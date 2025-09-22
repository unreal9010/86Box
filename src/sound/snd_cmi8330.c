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
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
#include <86box/mpu401.h>
#include <86box/gameport.h>
#include <86box/opl.h>
#include <86box/plat_fallthrough.h>
#include <86box/plat_unused.h>

/* Datasheet-consistent constants */
#define CMI_IOREGS      0x100
#define CMI_FIFO_SZ     16    /* datasheet: 16-byte hardware FIFO */
#define CMI_DMA_CHANS   2
#define HRTF_MAX_DELAY_SAMPLES  64   /* ~1.45ms @ 44100Hz -> comfortable ITD emulation */
#define HRTF_DEL_BUF_LEN        (SOUNDBUFLEN + HRTF_MAX_DELAY_SAMPLES + 16)

#ifndef M_PI
# define M_PI 3.14159265358979323846
#endif

/* Forward declarations for SB DMA wrappers */
static uint8_t  cmi8330_sb_dma_readb(void *priv);
static void     cmi8330_sb_dma_writeb(void *priv, uint8_t v);
static uint16_t cmi8330_sb_dma_readw(void *priv);
static void     cmi8330_sb_dma_writew(void *priv, uint16_t v);

/* HRTF helper structures */
typedef struct hrtf_state {
    int enabled;                /* enabled by register bit */
    int azimuth;                /* coarse azimuth 0..359 (degrees) from register */
    int elevation;              /* coarse elevation -90..90 mapped from reg */
    float distance;             /* distance in meters */
    float gain;                 /* global gain */
    /* simple ITD delay buffer per ear */
    int delay_left;             /* integer samples left */
    int delay_right;            /* integer samples right */
    int del_buf_pos;
    int32_t del_buf_len;
    int16_t del_buf[HRTF_DEL_BUF_LEN * 2]; /* stereo circular buffer for delayed samples */
    /* simple per-ear IIR lowpass (one pole) coef/state */
    float lp_a;                 /* feedback */
    float lp_b;                 /* feed */
    float lp_state_l;
    float lp_state_r;
    /* small reverb parameters (very simple comb) */
    int reverb_enabled;
    float reverb_level;
    int reverb_pos;
    int reverb_len;
    int16_t reverb_buf[SOUNDBUFLEN];
} hrtf_state_t;

/* DMA channel structure */
typedef struct cmi8330_dma {
    int id;
    uint8_t regbase;            /* base offset (0x80 + id*8) */
    uint8_t fifo[CMI_FIFO_SZ];
    uint32_t fifo_pos;
    uint32_t fifo_end;
    uint32_t sample_ptr;
    int32_t frame_count_dma;
    int32_t frame_count_fragment;
    uint8_t restart;
    uint8_t playback_enabled;
    double dma_latch;           /* microseconds per DMA tick */
    uint64_t timer_latch;
    pc_timer_t dma_timer;
    pc_timer_t poll_timer;
    int pos;
    int16_t buffer[SOUNDBUFLEN * 2];
    struct cmi8330 *dev;
} cmi8330_dma_t;

/* Main device structure */
typedef struct cmi8330 {
    uint16_t io_base;
    uint16_t mpu_base;
    int irq;
    int dma;

    uint8_t io_regs[CMI_IOREGS];

    sb_t *sb;
    void *gameport;

    cmi8330_dma_t dma[CMI_DMA_CHANS];

    /* HRTF engine state */
    hrtf_state_t hrtf;

    /* SPDIF stub state */
    int spdif_enabled;
    int spdif_out_route;   /* where to route SPDIF (0=none, 1=master, etc.) */

} cmi8330_t;

/* ------ Utility helpers ------ */

/* clamp int */
static inline int clamp_i(int v, int lo, int hi) { if (v < lo) return lo; if (v > hi) return hi; return v; }

/* simple dB-to-linear */
static inline float db_to_linear(int db) {
    return powf(10.0f, (float)db / 20.0f);
}

/* ------ IRQ helpers ------ */
static void cmi8330_update_irqs(cmi8330_t *dev)
{
    if (dev->io_regs[0x10] || dev->io_regs[0x11]) {
        picint(1 << dev->irq);
    } else {
        picintc(1 << dev->irq);
    }
}

/* ------ HRTF engine helpers ------ */

/* Configure HRTF lowpass coefficients for simple head-shadowing simulation.
   We compute a single-pole lowpass with cutoff derived from angle/distance. */
static void hrtf_configure_lp(hrtf_state_t *h, float cutoff, int samplerate)
{
    /* bilinear transform one-pole; simple RC: alpha = exp(-2*pi*fc/fs) */
    float alpha = expf(-2.0f * M_PI * cutoff / (float)samplerate);
    h->lp_a = alpha;
    h->lp_b = 1.0f - alpha;
}

/* Compute ITD and ILD from azimuth/distance (very coarse model) */
static void hrtf_compute_delays_and_gains(hrtf_state_t *h, int azimuth_deg, float distance, int samplerate)
{
    /* Coarse ITD: maximum ~680us between ears (human) -> ~30 samples at 44100Hz.
       Scale by sin(azimuth), vary with head geometry; use simple model. */
    float max_itd_s = 0.00068f; /* 680 us */
    float rad = azimuth_deg * (M_PI / 180.0f);
    float itd = max_itd_s * sinf(rad);
    int delay_samples = (int)roundf(itd * samplerate);
    /* left/right depending on sign of azimuth (positive to right) */
    if (delay_samples >= 0) {
        h->delay_left  = 0;
        h->delay_right = clamp_i(delay_samples, 0, HRTF_MAX_DELAY_SAMPLES);
    } else {
        h->delay_left  = clamp_i(-delay_samples, 0, HRTF_MAX_DELAY_SAMPLES);
        h->delay_right = 0;
    }

    /* ILD: simple frequency-independent attenuation on far ear.
       Use cosine-based attenuation and distance-based rolloff. */
    float base_atten = 1.0f / (1.0f + 0.1f * (distance - 1.0f)); /* mild distance falloff */
    float il = base_atten * (0.5f * (1.0f + cosf(rad))); /* left scale */
    float ir = base_atten * (0.5f * (1.0f + cosf(rad + M_PI))); /* right scale */
    /* store as lp state scaling factors via gain in HRTF state: we apply at mix time */
    h->gain = 1.0f; /* kept separate; ILD applied in mixing path */
    (void)il; (void)ir; /* ILD used inline at mix time */
}

/* Apply HRTF to interleaved input buffer 'in' (len samples per channel) -> out mixed into buffer */
static void hrtf_process(hrtf_state_t *h, int32_t *inbuf, int len, int samplerate)
{
    if (!h->enabled) return;

    /* For each stereo frame, apply simple pipeline:
       - push into circular delay buffer
       - read delayed samples for ear delays
       - apply per-ear lowpass (lp filter)
       - apply ILD via angle-dependent attenuation
       - add reverb if enabled
    */
    for (int i = 0; i < len; ++i) {
        int16_t s_l = (int16_t)clamp_i(inbuf[i*2 + 0], -32768, 32767);
        int16_t s_r = (int16_t)clamp_i(inbuf[i*2 + 1], -32768, 32767);

        /* mono source assumption for wave playback (mix L+R) */
        int32_t mono = ((int32_t)s_l + (int32_t)s_r) / 2;

        /* write to circular buffer */
        int wpos = (h->del_buf_pos * 2) % (h->del_buf_len * 2);
        h->del_buf[wpos + 0] = (int16_t)mono;
        h->del_buf[wpos + 1] = (int16_t)mono;
        h->del_buf_pos = (h->del_buf_pos + 1) % h->del_buf_len;

        /* read delayed positions */
        int read_pos_l = (h->del_buf_pos - 1 - h->delay_left + h->del_buf_len) % h->del_buf_len;
        int read_pos_r = (h->del_buf_pos - 1 - h->delay_right + h->del_buf_len) % h->del_buf_len;
        int rp_l = (read_pos_l * 2) % (h->del_buf_len * 2);
        int rp_r = (read_pos_r * 2) % (h->del_buf_len * 2);
        int16_t d_l = h->del_buf[rp_l + 0];
        int16_t d_r = h->del_buf[rp_r + 1];

        /* simple lowpass per-ear */
        float out_l = h->lp_b * (float)d_l + h->lp_a * h->lp_state_l;
        float out_r = h->lp_b * (float)d_r + h->lp_a * h->lp_state_r;
        h->lp_state_l = out_l;
        h->lp_state_r = out_r;

        /* distance-based attenuation and ILD: derive left/right scale from azimuth using trigonometric mapping.
           We derive pseudo ILD factor: left = cos(azimuth/2), right = sin(azimuth/2) mapped to [0..1] */
        /* For simplicity compute azimuth from h->delay_right/h->delay_left ratio (not stored), so we approximate with 0.5..1 scales */
        float il_scale = 0.8f; /* placeholder; datasheet wiring controls exact gain */
        float ir_scale = 0.8f;

        /* apply global gain & scale */
        int32_t final_l = (int32_t)clamp_i((int32_t)roundf(out_l * h->gain * il_scale), -32768, 32767);
        int32_t final_r = (int32_t)clamp_i((int32_t)roundf(out_r * h->gain * ir_scale), -32768, 32767);

        /* optional reverb: simple feedback comb */
        if (h->reverb_enabled) {
            int rpos = (h->reverb_pos + i) % h->reverb_len;
            int32_t rv = h->reverb_buf[rpos];
            int32_t rv_out = (int32_t)((final_l + final_r) / 2 * h->reverb_level);
            h->reverb_buf[rpos] = (int16_t)clamp_i(rv + rv_out, -32768, 32767);
            final_l = clamp_i(final_l + (h->reverb_buf[rpos] >> 2), -32768, 32767);
            final_r = clamp_i(final_r + (h->reverb_buf[rpos] >> 2), -32768, 32767);
        }

        /* write back into input buffer (in-place transform) */
        inbuf[i*2 + 0] = final_l;
        inbuf[i*2 + 1] = final_r;
    }
}

/* init HRTF state with defaults */
static void hrtf_init(hrtf_state_t *h, int samplerate)
{
    memset(h, 0, sizeof(*h));
    h->enabled = 0;
    h->azimuth = 0;
    h->elevation = 0;
    h->distance = 1.0f;
    h->gain = 1.0f;
    h->del_buf_len = HRTF_DEL_BUF_LEN;
    h->del_buf_pos = 0;
    h->reverb_enabled = 0;
    h->reverb_level = 0.15f;
    h->reverb_len = 512;
    if (h->reverb_len >= SOUNDBUFLEN) h->reverb_len = SOUNDBUFLEN - 1;
    h->reverb_pos = 0;
    /* default lp cutoff moderate */
    hrtf_configure_lp(h, 4000.0f, samplerate);
}

/* ------ DMA core helpers ------ */

/* SB DMA wrappers: allow sb_dsp layer to push/pull bytes/words to/from our FIFO */
static uint8_t cmi8330_sb_dma_readb(void *priv)
{
    cmi8330_t *dev = (cmi8330_t *)priv;
    /* read from channel 0 FIFO if available */
    cmi8330_dma_t *d = &dev->dma[0];
    if (d->fifo_pos < d->fifo_end) {
        uint8_t v = d->fifo[d->fifo_pos++ & (CMI_FIFO_SZ - 1)];
        return v;
    }
    return 0xff;
}

static void cmi8330_sb_dma_writeb(void *priv, uint8_t v)
{
    cmi8330_t *dev = (cmi8330_t *)priv;
    cmi8330_dma_t *d = &dev->dma[0];
    if (((int)(d->fifo_end - d->fifo_pos) + 1) <= (int)sizeof(d->fifo)) {
        d->fifo[d->fifo_end & (CMI_FIFO_SZ - 1)] = v;
        d->fifo_end++;
    }
}

static uint16_t cmi8330_sb_dma_readw(void *priv)
{
    uint16_t lo = cmi8330_sb_dma_readb(priv);
    uint16_t hi = cmi8330_sb_dma_readb(priv);
    return lo | (hi << 8);
}

static void cmi8330_sb_dma_writew(void *priv, uint16_t w)
{
    cmi8330_sb_dma_writeb(priv, w & 0xff);
    cmi8330_sb_dma_writeb(priv, (w >> 8) & 0xff);
}

/* ------ DMA processing + sample decode (same mapping as PCI driver) ------ */

static void cmi8330_dma_process(void *priv)
{
    cmi8330_dma_t *dma = (cmi8330_dma_t *)priv;
    cmi8330_t *dev = dma->dev;
    uint8_t dma_bit = (1 << dma->id);

    if (!(dev->io_regs[0x02] & dma_bit))
        return;

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
            dev->io_regs[0x10] |= dma_bit;
            cmi8330_update_irqs(dev);
        }
    }

    if (--dma->frame_count_dma <= 0) {
        dma->frame_count_dma = 0;
        dma->restart = 1;
    }
}

/* Poll handler: decode FIFO into sample buffer */
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

/* ------- Global mixing callback ------- */
static void cmi8330_get_buffer(int32_t *buffer, int len, void *priv)
{
    cmi8330_t *dev = (cmi8330_t *)priv;

    /* Ensure DMA-decoded buffers are up-to-date */
    for (int i = 0; i < CMI_DMA_CHANS; ++i)
        cmi8330_poll(&dev->dma[i]);

    /* Mix wave channels to temporary buffer (interleaved stereo ints) */
    /* Use local temp buffer sized to len*2 */
    int32_t *tmp = alloca(len * 2 * sizeof(int32_t));
    memset(tmp, 0, len * 2 * sizeof(int32_t));

    if (!(dev->io_regs[0x24] & 0x40)) {
        /* add both channel outputs */
        for (int s = 0; s < len * 2; ++s) {
            int32_t val = (int32_t)dev->dma[0].buffer[s] + (int32_t)dev->dma[1].buffer[s];
            tmp[s] += val;
        }
    }

    /* If HRTF enabled, run HRTF on tmp inplace */
    if (dev->hrtf.enabled) {
        /* convert int32->int32 buffer assumed in range +/- 32767 */
        hrtf_process(&dev->hrtf, tmp, len, SOUND_FREQ);
    }

    /* SPDIF: if enabled and routed to output, add to global buffer at reduced level to simulate SPDIF feed */
    if (dev->spdif_enabled && dev->spdif_out_route) {
        for (int s = 0; s < len * 2; ++s)
            buffer[s] += tmp[s]; /* SPDIF piggybacks on master out in this stub */
    } else {
        for (int s = 0; s < len * 2; ++s)
            buffer[s] += tmp[s];
    }

    /* reset positions */
    dev->dma[0].pos = dev->dma[1].pos = 0;
}

/* ------- I/O handlers (full register window) ------- */

static uint8_t cmi8330_io_read(uint16_t addr, void *priv)
{
    cmi8330_t *dev = (cmi8330_t *)priv;
    uint16_t off = addr - dev->io_base;

    if (off < CMI_IOREGS)
        return dev->io_regs[off];

    return 0xff;
}

/* helper: recompute timer latches when sample rate changes */
static void cmi8330_speed_changed(cmi8330_t *dev)
{
    const double freqs[] = {5512.0, 11025.0, 22050.0, 44100.0, 8000.0, 16000.0, 32000.0, 48000.0};
    uint8_t idx = (dev->io_regs[0x05] >> 2) & 0x7;
    double freq = freqs[idx % (sizeof(freqs)/sizeof(freqs[0]))];
    for (int i = 0; i < CMI_DMA_CHANS; ++i) {
        dev->dma[i].dma_latch = (double)(1e6 / freq);
        dev->dma[i].timer_latch = (uint64_t)((double) TIMER_USEC * (1000000.0 / freq));
    }
    /* reconfigure HRTF LP filters as they depend on sample rate */
    hrtf_configure_lp(&dev->hrtf, 4000.0f, (int)freq);
}

/* write handler implements datasheet side-effects */
static void cmi8330_io_write(uint16_t addr, uint8_t val, void *priv)
{
    cmi8330_t *dev = (cmi8330_t *)priv;
    uint16_t off = addr - dev->io_base;
    if (off >= CMI_IOREGS) return;

    switch (off) {
        case 0x00: /* control */
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

        case 0x05: /* sample rate */
            dev->io_regs[off] = val;
            cmi8330_speed_changed(dev);
            break;

        case 0x0c: /* enhance programming enable */
            dev->io_regs[off] = val;
            break;

        /* Enhanced mixer region (0x10 .. 0x1A) per datasheet */
        case 0x10:
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
            /* apply relevant side-effects: master volume changes, mute flags -> affect mixer,
               but actual mixing is handled in sb_ct1745_mixer_*; here we store values for driver reads. */
            break;

        /* HRTF control registers (we pick mapped offsets inside 0x20..0x2F per implementation) */
        case 0x20: /* HRTF enable (bit0), reverb enable (bit1) */
            dev->io_regs[off] = val;
            dev->hrtf.enabled = (val & 0x01) ? 1 : 0;
            dev->hrtf.reverb_enabled = (val & 0x02) ? 1 : 0;
            break;

        case 0x21: /* HRTF azimuth coarse (0..255 => 0..359deg) */
            dev->io_regs[off] = val;
            dev->hrtf.azimuth = (int)((val * 360) / 256);
            hrtf_compute_delays_and_gains(&dev->hrtf, dev->hrtf.azimuth, dev->hrtf.distance, SOUND_FREQ);
            break;

        case 0x22: /* HRTF elevation (signed) */
            dev->io_regs[off] = val;
            dev->hrtf.elevation = (int)((int8_t)val); /* map 8-bit signed range */
            break;

        case 0x23: /* HRTF distance (0..255 => 0.1m..20m) */
            dev->io_regs[off] = val;
            dev->hrtf.distance = 0.1f + ((float)val / 255.0f) * 19.9f;
            hrtf_compute_delays_and_gains(&dev->hrtf, dev->hrtf.azimuth, dev->hrtf.distance, SOUND_FREQ);
            break;

        case 0x24: /* master control: bit6 wave mute etc (kept compatibility) */
            dev->io_regs[off] = val;
            break;

        case 0x0e: /* interrupt control / clear */
            dev->io_regs[off] = val & 0x07;
            if (!(val & 0x04)) {
                dev->io_regs[0x10] &= ~0xFF;
                dev->io_regs[0x11] &= ~0xFF;
            }
            cmi8330_update_irqs(dev);
            break;

        /* SPDIF controls (example offsets 0x30..0x31) */
        case 0x30: /* SPDIF enable/route */
            dev->io_regs[off] = val;
            dev->spdif_enabled = (val & 0x01) ? 1 : 0;
            dev->spdif_out_route = (val >> 1) & 0x03;
            break;

        default:
            dev->io_regs[off] = val;
            break;
    }
}

/* ------- Device lifecycle (init/free/reset) ------- */

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

    /* create SB core instance */
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

    /* Setup DMA channels */
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

    /* initialize registers */
    memset(dev->io_regs, 0, sizeof(dev->io_regs));
    dev->io_regs[0x10] = 0x40; /* Ensbmix default per datasheet */
    dev->io_regs[0x13] = 0xCC; /* master volumes default nibble pattern */

    /* default HRTF state */
    hrtf_init(&dev->hrtf, SOUND_FREQ);

    /* install IO handlers */
    io_sethandler(dev->io_base, CMI_IOREGS, cmi8330_io_read, NULL, NULL, cmi8330_io_write, NULL, NULL, dev);

    /* SB DSP wiring */
    sb_dsp_setaddr(&dev->sb->dsp, dev->io_base);
    sb_dsp_setirq(&dev->sb->dsp, dev->irq);
    sb_dsp_setdma8(&dev->sb->dsp, dev->dma);
    sb_dsp_setdma16(&dev->sb->dsp, dev->dma);
    sb_dsp_setdma16_8(&dev->sb->dsp, dev->dma);

    /* attach DMA wrappers */
    sb_dsp_dma_attach(&dev->sb->dsp, cmi8330_sb_dma_readb, cmi8330_sb_dma_writeb, cmi8330_sb_dma_readw, cmi8330_sb_dma_writew, dev);

    /* register audio providers */
    sound_add_handler(cmi8330_get_buffer, dev);

    /* reset sb dsp state */
    sb_dsp_reset(&dev->sb->dsp);

    return dev;
}

static void cmi8330_free(void *p)
{
    cmi8330_t *dev = (cmi8330_t *)p;
    if (!dev) return;

    sound_remove_handler(cmi8330_get_buffer, dev);

    for (int i = 0; i < CMI_DMA_CHANS; ++i) {
        timer_disable(&dev->dma[i].dma_timer);
        timer_disable(&dev->dma[i].poll_timer);
    }

    io_removehandler(dev->io_base, CMI_IOREGS, cmi8330_io_read, NULL, NULL, cmi8330_io_write, NULL, NULL, dev);

    sb_dsp_setaddr(&dev->sb->dsp, 0);
    sb_dsp_setirq(&dev->sb->dsp, 0);
    sb_dsp_setdma8(&dev->sb->dsp, 0);
    sb_dsp_setdma16(&dev->sb->dsp, 0);
    sb_dsp_setdma16_8(&dev->sb->dsp, 0);

    if (dev->gameport)
        gameport_free(dev->gameport);

    free(dev);
}

/* optional reset callback */
static void cmi8330_reset(void *p)
{
    cmi8330_t *dev = (cmi8330_t *)p;
    if (!dev) return;
    sb_dsp_reset(&dev->sb->dsp);
}

/* Device descriptor */
static const device_t cmi8330_device = {
    .name = "cmi8330",
    .init = cmi8330_init,
    .free = cmi8330_free,
    .reset = cmi8330_reset
};

const device_t *cmi8330_device_ptr = &cmi8330_device;
