/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          C-Media CMI8330 ISA audio device emulation.
 *
 * Authors: Sarah Walker, <https://pcem-emulator.co.uk/>
 *          TheCollector1995, <mariogplayer@gmail.com>
 *          RichardG, <richardg867@gmail.com>
 *          unreal9010
 *
 *          Copyright 2008-2020 Sarah Walker.
 *          Copyright 2018-2020 TheCollector1995.
 *          Copyright 2021-2025 RichardG.
 *          Copyright 2025 unreal9010.
 */
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define HAVE_STDARG_H

#include <86box/86box.h>
#include <86box/device.h>
#include <86box/io.h>
#include <86box/mem.h>
#include <86box/pic.h>
#include <86box/timer.h>
#include <86box/dma.h>
#include <86box/sound.h>
#include <86box/snd_sb.h>
#include <86box/snd_sb_dsp.h>
#include <86box/snd_opl.h>
#include <86box/midi.h>
#include <86box/gameport.h>
#include <86box/isapnp.h>
#include <86box/plat_fallthrough.h>
#include <86box/plat_unused.h>

/* Datasheet-driven constants */
#define CMI_IOREGS      0x100
#define CMI_FIFO_SZ     16
#define CMI_DMA_CHANS   2

/* Forward declarations (device entry at bottom) */
static void *cmi8330_init(const device_t *info);
static void  cmi8330_close(void *priv);
static void  cmi8330_reset(void *priv);

/* Minimal HRTF state (lightweight perceptual stub) */
typedef struct hrtf_state {
    int enabled;
    int azimuth;      /* 0..359 degrees */
    int elevation;    /* -90..90 */
    float distance;   /* in meters */
    float gain;
    int delay_left;   /* integer samples */
    int delay_right;
    int del_buf_pos;
    int del_buf_len;
    int16_t del_buf[1024]; /* small circular buffer */
    float lp_a, lp_b;
    float lp_state_l, lp_state_r;
    int reverb_enabled;
    float reverb_level;
    int16_t reverb_buf[1024];
    int reverb_pos;
} hrtf_state_t;

/* DMA channel */
typedef struct cmi8330_dma {
    int id;
    uint8_t regbase;            /* register base (0x80 + id*8) */
    uint8_t fifo[CMI_FIFO_SZ];
    uint32_t fifo_pos;
    uint32_t fifo_end;
    uint32_t sample_ptr;
    int32_t frame_count_dma;
    int32_t frame_count_fragment;
    uint8_t restart;
    uint8_t playback_enabled;
    double dma_latch;
    uint64_t timer_latch;
    pc_timer_t dma_timer;
    pc_timer_t poll_timer;
    int pos;
    int16_t buffer[SOUNDBUFLEN * 2];
    struct cmi8330 *dev;        /* back pointer */
} cmi8330_dma_t;

/* Main device state */
typedef struct cmi8330 {
    uint16_t io_base;
    uint16_t mpu_base;
    int irq;
    int dma;

    uint8_t io_regs[CMI_IOREGS];

    sb_t *sb;
    void *gameport;

    cmi8330_dma_t dma[CMI_DMA_CHANS];

    hrtf_state_t hrtf;
    int spdif_enabled;
    int spdif_route;
} cmi8330_t;

/* ---------- Small helpers ---------- */

static inline int clamp_i(int v, int lo, int hi) { if (v < lo) return lo; if (v > hi) return hi; return v; }

/* Configure a one-pole lowpass for HRTF (simple head-shadowing proxy) */
static void hrtf_configure_lp(hrtf_state_t *h, float cutoff, int samplerate)
{
    float alpha = expf(-2.0f * M_PI * cutoff / (float)samplerate);
    h->lp_a = alpha;
    h->lp_b = 1.0f - alpha;
}

/* Compute coarse ITD and set simple gain/delay parameters */
static void hrtf_compute_basic(hrtf_state_t *h, int azimuth_deg, float distance, int samplerate)
{
    float max_itd_s = 0.00068f; /* ~680us */
    float rad = azimuth_deg * (M_PI / 180.0f);
    float itd = max_itd_s * sinf(rad);
    int delay_samples = (int)roundf(itd * samplerate);
    if (delay_samples >= 0) {
        h->delay_left = 0;
        h->delay_right = clamp_i(delay_samples, 0, 64);
    } else {
        h->delay_left = clamp_i(-delay_samples, 0, 64);
        h->delay_right = 0;
    }
    /* coarse distance -> gain */
    h->gain = 1.0f / (1.0f + 0.1f * (distance - 1.0f));
}

/* init HRTF */
static void hrtf_init(hrtf_state_t *h, int samplerate)
{
    memset(h, 0, sizeof(*h));
    h->enabled = 0;
    h->azimuth = 0;
    h->elevation = 0;
    h->distance = 1.0f;
    h->gain = 1.0f;
    h->del_buf_len = (int)(sizeof(h->del_buf)/sizeof(h->del_buf[0]));
    h->del_buf_pos = 0;
    h->reverb_len = 512; /* small */
    hrtf_configure_lp(h, 4000.0f, samplerate);
    hrtf_compute_basic(h, h->azimuth, h->distance, samplerate);
}

/* ---------- SB DSP DMA wrappers (must match sb_dsp_dma_attach signatures) ---------- */

/* Return DMA_NODATA (-1) if there is no data; otherwise return byte/word value. */
static int cmi8330_sb_dma_readb(void *priv)
{
    cmi8330_t *dev = (cmi8330_t *)priv;
    cmi8330_dma_t *d = &dev->dma[0]; /* legacy SB uses channel 0 FIFO */
    if (d->fifo_pos < d->fifo_end) {
        int v = d->fifo[d->fifo_pos++ & (CMI_FIFO_SZ - 1)];
        return v & 0xff;
    }
    return DMA_NODATA;
}

static int cmi8330_sb_dma_readw(void *priv)
{
    int lo = cmi8330_sb_dma_readb(priv);
    if (lo == DMA_NODATA) return DMA_NODATA;
    int hi = cmi8330_sb_dma_readb(priv);
    if (hi == DMA_NODATA) return DMA_NODATA;
    return (lo & 0xff) | ((hi & 0xff) << 8);
}

/* Write callbacks — return 0 on success, non-zero on error (match other drivers) */
static int cmi8330_sb_dma_writeb(void *priv, uint8_t val)
{
    cmi8330_t *dev = (cmi8330_t *)priv;
    cmi8330_dma_t *d = &dev->dma[0];
    if (((int)(d->fifo_end - d->fifo_pos) + 1) <= (int)sizeof(d->fifo)) {
        d->fifo[d->fifo_end & (CMI_FIFO_SZ - 1)] = val;
        d->fifo_end++;
        return 0;
    }
    return 1;
}

static int cmi8330_sb_dma_writew(void *priv, uint16_t val)
{
    if (cmi8330_sb_dma_writeb(priv, val & 0xff)) return 1;
    if (cmi8330_sb_dma_writeb(priv, (val >> 8) & 0xff)) return 1;
    return 0;
}

/* ---------- DMA processing and polling ---------- */

static void cmi8330_dma_process(void *priv)
{
    cmi8330_dma_t *dma = (cmi8330_dma_t *)priv;
    cmi8330_t *dev = dma->dev;
    uint8_t dma_bit = (1 << dma->id);

    /* DMA enable register at 0x02 */
    if (!(dev->io_regs[0x02] & dma_bit))
        return;

    /* re-arm timer */
    timer_on_auto(&dma->dma_timer, dma->dma_latch);

    if (dma->restart) {
        dma->restart = 0;
        uint32_t p = dev->io_regs[dma->regbase + 0] |
                     (dev->io_regs[dma->regbase + 1] << 8) |
                     (dev->io_regs[dma->regbase + 2] << 16) |
                     (dev->io_regs[dma->regbase + 3] << 24);
        dma->sample_ptr = p;
        dma->frame_count_fragment = (dev->io_regs[dma->regbase + 4] | (dev->io_regs[dma->regbase + 5] << 8)) + 1;
        dma->frame_count_dma = (dev->io_regs[dma->regbase + 6] | (dev->io_regs[dma->regbase + 7] << 8)) + 1;
    }

    int writeback = (dev->io_regs[0x00] >> dma->id) & 1;

    if (writeback) {
        if ((int)(dma->fifo_end - dma->fifo_pos) >= 4) {
            uint32_t v;
            uint32_t idx = dma->fifo_pos & (CMI_FIFO_SZ - 1);
            memcpy(&v, &dma->fifo[idx], 4);
            mem_writel_phys(dma->sample_ptr, v);
            dma->fifo_pos += 4;
            dma->sample_ptr += 4;
        }
    } else {
        if (((int)(dma->fifo_end - dma->fifo_pos) + 4) <= (int)sizeof(dma->fifo)) {
            uint32_t v = mem_readl_phys(dma->sample_ptr);
            uint32_t idx = dma->fifo_end & (CMI_FIFO_SZ - 1);
            memcpy(&dma->fifo[idx], &v, 4);
            dma->fifo_end += 4;
            dma->sample_ptr += 4;
        }
    }

    if (--dma->frame_count_fragment <= 0) {
        dma->frame_count_fragment = (dev->io_regs[dma->regbase + 4] | (dev->io_regs[dma->regbase + 5] << 8)) + 1;
        if (dev->io_regs[0x0e] & dma_bit) {
            dev->io_regs[0x10] |= dma_bit; /* fragment interrupt flag region */
            /* ISR raise */
            picint(1 << dev->irq);
        }
    }

    if (--dma->frame_count_dma <= 0) {
        dma->frame_count_dma = 0;
        dma->restart = 1;
    }
}

/* Decode FIFO into sample buffer; runs at sample rate */
static void cmi8330_poll(void *priv)
{
    cmi8330_dma_t *dma = (cmi8330_dma_t *)priv;
    cmi8330_t *dev = dma->dev;

    if (dma->playback_enabled)
        timer_advance_u64(&dma->poll_timer, dma->timer_latch);

    for (; dma->pos < sound_pos_global; dma->pos++) {
        int16_t left = 0, right = 0;
        int fmt = (dev->io_regs[0x08] >> (dma->id << 1)) & 0x3;

        if (fmt == 0) {
            if ((int)(dma->fifo_end - dma->fifo_pos) >= 1) {
                uint8_t v = dma->fifo[dma->fifo_pos++ & (CMI_FIFO_SZ - 1)];
                left = right = ((int)v ^ 0x80) << 8;
            }
        } else if (fmt == 1) {
            if ((int)(dma->fifo_end - dma->fifo_pos) >= 2) {
                uint8_t vl = dma->fifo[dma->fifo_pos++ & (CMI_FIFO_SZ - 1)];
                uint8_t vr = dma->fifo[dma->fifo_pos++ & (CMI_FIFO_SZ - 1)];
                left = ((int)vl ^ 0x80) << 8;
                right = ((int)vr ^ 0x80) << 8;
            }
        } else {
            if ((int)(dma->fifo_end - dma->fifo_pos) >= 4) {
                uint8_t b0 = dma->fifo[dma->fifo_pos++ & (CMI_FIFO_SZ - 1)];
                uint8_t b1 = dma->fifo[dma->fifo_pos++ & (CMI_FIFO_SZ - 1)];
                uint8_t b2 = dma->fifo[dma->fifo_pos++ & (CMI_FIFO_SZ - 1)];
                uint8_t b3 = dma->fifo[dma->fifo_pos++ & (CMI_FIFO_SZ - 1)];
                left = (int16_t)(b0 | (b1 << 8));
                right = (int16_t)(b2 | (b3 << 8));
            }
        }

        dma->buffer[dma->pos * 2 + 0] = left;
        dma->buffer[dma->pos * 2 + 1] = right;
    }
}

/* Very small HRTF processor applied to interleaved int32 buffer */
static void hrtf_process(hrtf_state_t *h, int32_t *buf, int len, int samplerate)
{
    if (!h->enabled) return;

    for (int i = 0; i < len; ++i) {
        int32_t in_l = clamp_i(buf[i*2 + 0], -32768, 32767);
        int32_t in_r = clamp_i(buf[i*2 + 1], -32768, 32767);
        int32_t mono = (in_l + in_r) / 2;

        /* circular delay write */
        h->del_buf[h->del_buf_pos] = (int16_t)mono;
        h->del_buf_pos = (h->del_buf_pos + 1) % h->del_buf_len;

        int read_l = (h->del_buf_pos - 1 - h->delay_left + h->del_buf_len) % h->del_buf_len;
        int read_r = (h->del_buf_pos - 1 - h->delay_right + h->del_buf_len) % h->del_buf_len;
        int16_t d_l = h->del_buf[read_l];
        int16_t d_r = h->del_buf[read_r];

        float out_l = h->lp_b * (float)d_l + h->lp_a * h->lp_state_l;
        float out_r = h->lp_b * (float)d_r + h->lp_a * h->lp_state_r;
        h->lp_state_l = out_l;
        h->lp_state_r = out_r;

        float il = 0.9f, ir = 0.9f; /* simple ILD stub */
        int32_t final_l = clamp_i((int)roundf(out_l * h->gain * il), -32768, 32767);
        int32_t final_r = clamp_i((int)roundf(out_r * h->gain * ir), -32768, 32767);

        if (h->reverb_enabled) {
            int rp = (h->reverb_pos++) % (int)(sizeof(h->reverb_buf)/sizeof(h->reverb_buf[0]));
            int32_t rv = h->reverb_buf[rp];
            int32_t add = (int32_t)((final_l + final_r) / 4 * h->reverb_level);
            h->reverb_buf[rp] = clamp_i(rv + add, -32768, 32767);
            final_l = clamp_i(final_l + (h->reverb_buf[rp] >> 3), -32768, 32767);
            final_r = clamp_i(final_r + (h->reverb_buf[rp] >> 3), -32768, 32767);
        }

        buf[i*2 + 0] = final_l;
        buf[i*2 + 1] = final_r;
    }
}

/* ---------- Mixing callback ---------- */

static void cmi8330_get_buffer(int32_t *buffer, int len, void *priv)
{
    cmi8330_t *dev = (cmi8330_t *)priv;

    /* ensure FIFO decode up-to-date */
    for (int i = 0; i < CMI_DMA_CHANS; ++i)
        cmi8330_poll(&dev->dma[i]);

    /* temporary mixing buffer */
    int32_t tmp[len * 2];
    memset(tmp, 0, sizeof(tmp));

    /* master mute bit at 0x24 bit6 (spec/implementation dependent) */
    if (!(dev->io_regs[0x24] & 0x40)) {
        for (int s = 0; s < len * 2; ++s)
            tmp[s] += (int32_t)dev->dma[0].buffer[s] + (int32_t)dev->dma[1].buffer[s];
    }

    /* apply HRTF if enabled */
    if (dev->hrtf.enabled)
        hrtf_process(&dev->hrtf, tmp, len, SOUND_FREQ);

    /* SPDIF stub: route to master output if enabled (simple mix) */
    for (int s = 0; s < len * 2; ++s)
        buffer[s] += tmp[s];

    /* reset per-frame positions */
    dev->dma[0].pos = dev->dma[1].pos = 0;
}

/* ---------- I/O window handlers ---------- */

static uint8_t cmi8330_io_read(uint16_t addr, void *priv)
{
    cmi8330_t *dev = (cmi8330_t *)priv;
    uint16_t off = addr - dev->io_base;
    if (off < CMI_IOREGS)
        return dev->io_regs[off];
    return 0xff;
}

static void cmi8330_speed_changed(cmi8330_t *dev)
{
    const double freqs[] = {5512.0, 11025.0, 22050.0, 44100.0, 8000.0, 16000.0, 32000.0, 48000.0};
    uint8_t idx = (dev->io_regs[0x05] >> 2) & 0x7;
    double freq = freqs[idx % (sizeof(freqs)/sizeof(freqs[0]))];
    for (int i = 0; i < CMI_DMA_CHANS; ++i) {
        dev->dma[i].dma_latch = (double)(1e6 / freq);
        dev->dma[i].timer_latch = (uint64_t)((double) TIMER_USEC * (1000000.0 / freq));
    }
    hrtf_configure_lp(&dev->hrtf, 4000.0f, (int)freq);
}

static void cmi8330_io_write(uint16_t addr, uint8_t val, void *priv)
{
    cmi8330_t *dev = (cmi8330_t *)priv;
    uint16_t off = addr - dev->io_base;
    if (off >= CMI_IOREGS) return;

    switch (off) {
        case 0x00: /* control: direction & start/stop */
            dev->io_regs[off] = val;
            for (int i = 0; i < CMI_DMA_CHANS; ++i) {
                if (val & (1 << i)) {
                    dev->dma[i].restart = 1;
                    timer_on_auto(&dev->dma[i].dma_timer, dev->dma[i].dma_latch);
                    timer_on_auto(&dev->dma[i].poll_timer, dev->dma[i].timer_latch);
                } else {
                    timer_disable(&dev->dma[i].dma_timer);
                    timer_disable(&dev->dma[i].poll_timer);
                }
            }
            break;

        case 0x02: /* DMA enable */
            dev->io_regs[off] = val;
            for (int i = 0; i < CMI_DMA_CHANS; ++i)
                dev->dma[i].playback_enabled = (val & (1 << i)) ? 1 : 0;
            break;

        case 0x04: /* remap / gameport */
            dev->io_regs[off] = val;
            if (dev->gameport) {
                if (val & 0x02) gameport_remap(dev->gameport, 0x200);
                else gameport_remap(dev->gameport, 0);
            }
            break;

        case 0x05: /* sample rate / clock */
            dev->io_regs[off] = val;
            cmi8330_speed_changed(dev);
            break;

        case 0x0c: /* enhance programming enable */
            dev->io_regs[off] = val;
            break;

        case 0x10: /* enhanced mixer region (record routing / flags) */
        case 0x11:
        case 0x12:
        case 0x13:
        case 0x14:
        case 0x15:
        case 0x16:
        case 0x17:
        case 0x18:
        case 0x19:
        case 0x1a:
            dev->io_regs[off] = val;
            break;

        case 0x20: /* HRTF control (example mapping) */
            dev->io_regs[off] = val;
            dev->hrtf.enabled = (val & 0x01) ? 1 : 0;
            dev->hrtf.reverb_enabled = (val & 0x02) ? 1 : 0;
            break;

        case 0x21: /* HRTF azimuth */
            dev->io_regs[off] = val;
            dev->hrtf.azimuth = (int)((val * 360) / 256);
            hrtf_compute_basic(&dev->hrtf, dev->hrtf.azimuth, dev->hrtf.distance, SOUND_FREQ);
            break;

        case 0x22: /* HRTF elevation */
            dev->io_regs[off] = val;
            dev->hrtf.elevation = (int)((int8_t)val);
            break;

        case 0x23: /* HRTF distance */
            dev->io_regs[off] = val;
            dev->hrtf.distance = 0.1f + ((float)val / 255.0f) * 19.9f;
            hrtf_compute_basic(&dev->hrtf, dev->hrtf.azimuth, dev->hrtf.distance, SOUND_FREQ);
            break;

        case 0x24: /* master control (mute bits etc) */
            dev->io_regs[off] = val;
            break;

        case 0x0e: /* interrupt control/clear */
            dev->io_regs[off] = val & 0x07;
            if (!(val & 0x04)) {
                dev->io_regs[0x10] = 0;
                dev->io_regs[0x11] = 0;
            }
            break;

        case 0x30: /* SPDIF control */
            dev->io_regs[off] = val;
            dev->spdif_enabled = (val & 0x01) ? 1 : 0;
            dev->spdif_route = (val >> 1) & 0x03;
            break;

        default:
            dev->io_regs[off] = val;
            break;
    }
}

/* ---------- Lifecycle ---------- */

static void *cmi8330_init(const device_t *info)
{
    cmi8330_t *dev = calloc(1, sizeof(cmi8330_t));
    if (!dev) return NULL;

    dev->io_base = device_get_config_hex16("base");
    if (!dev->io_base) dev->io_base = 0x220;
    dev->mpu_base = device_get_config_hex16("base401");
    dev->irq = device_get_config_int("irq");
    if (!dev->irq) dev->irq = 5;
    dev->dma = device_get_config_int("dma");
    if (!dev->dma) dev->dma = 1;

    /* Create Sound Blaster core instance (same pattern used in other drivers) */
    dev->sb = device_add_inst(device_get_config_int("receive_input") ? &sb_16_compat_device : &sb_16_compat_nompu_device, 1);
    if (!dev->sb) {
        free(dev);
        return NULL;
    }

    dev->sb->opl_enabled = 1;

    if (dev->mpu_base && dev->sb->mpu)
        mpu401_change_addr(dev->sb->mpu, dev->mpu_base);

    if (device_get_config_int("gameport")) {
        dev->gameport = gameport_add(&gameport_pnp_device);
        dev->sb->gameport_addr = 0x200;
        gameport_remap(dev->gameport, dev->sb->gameport_addr);
    }

    /* Setup DMA channels and timers */
    for (int i = 0; i < CMI_DMA_CHANS; ++i) {
        cmi8330_dma_t *dma = &dev->dma[i];
        dma->id = i;
        dma->regbase = 0x80 + (i << 3);
        dma->fifo_pos = dma->fifo_end = 0;
        dma->restart = 1;
        dma->dev = dev;
        dma->dma_latch = 1e6 / 44100.0;
        dma->timer_latch = (uint64_t)((double) TIMER_USEC * (1000000.0 / 44100.0));
        timer_add(&dma->dma_timer, cmi8330_dma_process, dma, 0);
        timer_add(&dma->poll_timer, cmi8330_poll, dma, 0);
    }

    /* clear registers and set a few sensible defaults from datasheet */
    memset(dev->io_regs, 0, sizeof(dev->io_regs));
    dev->io_regs[0x10] = 0x40; /* Ensbmix default bit per datasheet */
    dev->io_regs[0x13] = 0xCC; /* master volume nibble defaults (if documented) */

    /* init HRTF */
    hrtf_init(&dev->hrtf, SOUND_FREQ);

    /* install IO handlers for 256-byte window at base */
    io_sethandler(dev->io_base, CMI_IOREGS, cmi8330_io_read, NULL, NULL, cmi8330_io_write, NULL, NULL, dev);

    /* SB DSP wiring */
    sb_dsp_setaddr(&dev->sb->dsp, dev->io_base);
    sb_dsp_setirq(&dev->sb->dsp, dev->irq);
    sb_dsp_setdma8(&dev->sb->dsp, dev->dma);
    sb_dsp_setdma16(&dev->sb->dsp, dev->dma);
    sb_dsp_setdma16_8(&dev->sb->dsp, dev->dma);

    /* attach DMA callbacks (must match signatures in snd_sb_dsp.c) */
    sb_dsp_dma_attach(&dev->sb->dsp,
                      cmi8330_sb_dma_readb, cmi8330_sb_dma_readw,
                      cmi8330_sb_dma_writeb, cmi8330_sb_dma_writew,
                      dev);

    /* register audio producer */
    sound_add_handler(cmi8330_get_buffer, dev);

    /* reset SB DSP internal state */
    sb_dsp_reset(&dev->sb->dsp);

    return dev;
}

static void cmi8330_close(void *p)
{
    cmi8330_t *dev = (cmi8330_t *)p;
    if (!dev) return;

    for (int i = 0; i < CMI_DMA_CHANS; ++i) {
        timer_disable(&dev->dma[i].dma_timer);
        timer_disable(&dev->dma[i].poll_timer);
    }

    io_removehandler(dev->io_base, CMI_IOREGS, cmi8330_io_read, NULL, NULL, cmi8330_io_write, NULL, NULL, dev);

    /* detach SB DSP wiring */
    sb_dsp_setaddr(&dev->sb->dsp, 0);
    sb_dsp_setirq(&dev->sb->dsp, 0);
    sb_dsp_setdma8(&dev->sb->dsp, 0);
    sb_dsp_setdma16(&dev->sb->dsp, 0);
    sb_dsp_setdma16_8(&dev->sb->dsp, 0);

    /* gameport cleanup is handled elsewhere in the core; free our container */
    free(dev);
}

static void cmi8330_reset(void *p)
{
    cmi8330_t *dev = (cmi8330_t *)p;
    if (!dev) return;
    sb_dsp_reset(&dev->sb->dsp);
}

const device_t cmi8330_device = {
    .name          = "C-Media CMI8330",
    .internal_name = "cmi8330",
    .flags         = DEVICE_ISA,
    .local         = 2,               
    .init          = cmi8330_init,    
    .close         = cmi8330_free,    
    .reset         = NULL,            
    .available     = cmi8330_available,   
    .speed_changed = NULL,            
    .force_redraw  = NULL,
    .config        = cmi8330_config   
};

const device_t cmi8330_device = {
    .name          = "C-Media CMI8330 (onboard)",
    .internal_name = "cmi8330_onboard",
    .flags         = DEVICE_ISA,
    .local         = 2,               
    .init          = cmi8330_init,    
    .close         = cmi8330_free,    
    .reset         = NULL,            
    .available     = cmi8330_available,   
    .speed_changed = NULL,            
    .force_redraw  = NULL,
    .config        = cmi8330_config   
};
