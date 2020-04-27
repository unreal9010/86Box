/*
 * 86Box	A hypervisor and IBM PC system emulator that specializes in
 *		running old operating systems and software designed for IBM
 *		PC systems and compatibles from 1981 through fairly recent
 *		system designs based on the PCI bus.
 *
 *		This file is part of the 86Box distribution.
 *
 *		Memory handling and MMU.
 *
 * NOTE:	Experimenting with dynamically allocated lookup tables;
 *		the DYNAMIC_TABLES=1 enables this. Will eventually go
 *		away, either way...
 *
 *
 *
 * Authors:	Sarah Walker, <tommowalker@tommowalker.co.uk>
 *		Miran Grca, <mgrca8@gmail.com>
 *		Fred N. van Kempen, <decwiz@yahoo.com>
 *
 *		Copyright 2008-2020 Sarah Walker.
 *		Copyright 2016-2020 Miran Grca.
 *		Copyright 2017-2020 Fred N. van Kempen.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <wchar.h>
#define HAVE_STDARG_H
#include <86box/86box.h>
#include "cpu.h"
#include "x86_ops.h"
#include "x86.h"
#include <86box/machine.h>
#include <86box/m_xt_xi8088.h>
#include <86box/config.h>
#include <86box/io.h>
#include <86box/mem.h>
#include <86box/rom.h>
#ifdef USE_DYNAREC
# include "codegen_public.h"
#else
#ifdef USE_NEW_DYNAREC
# define PAGE_MASK_SHIFT	6
#else
# define PAGE_MASK_INDEX_MASK	3
# define PAGE_MASK_INDEX_SHIFT	10
# define PAGE_MASK_SHIFT	4
#endif
# define PAGE_MASK_MASK		63
#endif


#define FIXME			0
#define DYNAMIC_TABLES		0		/* experimental */


mem_mapping_t		base_mapping,
			ram_low_mapping,	/* 0..640K mapping */
#if 1
			ram_mid_mapping,
#endif
			ram_remapped_mapping,	/* 640..1024K mapping */
			ram_high_mapping,	/* 1024K+ mapping */
			ram_remapped_mapping,
			ram_split_mapping,
			bios_mapping,
			bios_high_mapping;

page_t			*pages,			/* RAM page table */
			**page_lookup;		/* pagetable lookup */
uint32_t		pages_sz;		/* #pages in table */

uint8_t			*ram;			/* the virtual RAM */
uint32_t		rammask;

uint8_t			*rom;			/* the virtual ROM */
uint32_t		biosmask, biosaddr;

uint32_t		pccache;
uint8_t			*pccache2;

int			readlnext;
int			readlookup[256],
			readlookupp[256];
uintptr_t		*readlookup2;
int			writelnext;
int			writelookup[256],
			writelookupp[256];
uintptr_t		*writelookup2;

uint32_t		mem_logical_addr;

int			shadowbios = 0,
			shadowbios_write;
int			readlnum = 0,
			writelnum = 0;
int			cachesize = 256;

uint32_t		get_phys_virt,
			get_phys_phys;

int			mem_a20_key = 0,
			mem_a20_alt = 0,
			mem_a20_state = 0;

int			mmuflush = 0;
int			mmu_perm = 4;

uint64_t		*byte_dirty_mask;
uint64_t		*byte_code_present_mask;

uint32_t		purgable_page_list_head = 0;
int			purgeable_page_count = 0;


/* FIXME: re-do this with a 'mem_ops' struct. */
static mem_mapping_t	*read_mapping[MEM_MAPPINGS_NO];
static mem_mapping_t	*write_mapping[MEM_MAPPINGS_NO];
static uint8_t		*_mem_exec[MEM_MAPPINGS_NO];
static int		_mem_state[MEM_MAPPINGS_NO];

#if FIXME
#if (MEM_GRANULARITY_BITS >= 12)
static uint8_t		ff_array[MEM_GRANULARITY_SIZE];
#else
static uint8_t		ff_array[4096];			/* Must be at least one page. */
#endif
#else
static uint8_t		ff_pccache[4] = { 0xff, 0xff, 0xff, 0xff };
#endif


#define USE_PHYS_EXEC


#ifdef ENABLE_MEM_LOG
int mem_do_log = ENABLE_MEM_LOG;


static void
mem_log(const char *fmt, ...)
{
    va_list ap;

    if (mem_do_log) {
	va_start(ap, fmt);
	pclog_ex(fmt, ap);
	va_end(ap);
    }
}
#else
#define mem_log(fmt, ...)
#endif


int
mem_addr_is_ram(uint32_t addr)
{
    mem_mapping_t *mapping = read_mapping[addr >> MEM_GRANULARITY_BITS];

    return (mapping == &ram_low_mapping) || (mapping == &ram_high_mapping) || (mapping == &ram_mid_mapping) || (mapping == &ram_remapped_mapping);
}


void
resetreadlookup(void)
{
    int c;

    /* This is NULL after app startup, when mem_init() has not yet run. */
#if DYNAMIC_TABLES
mem_log("MEM: reset_lookup: pages=%08lx, lookup=%08lx, pages_sz=%i\n", pages, page_lookup, pages_sz);
#endif

    /* Initialize the page lookup table. */
#if DYNAMIC_TABLES
    memset(page_lookup, 0x00, pages_sz*sizeof(page_t *));
#else
    memset(page_lookup, 0x00, (1<<20)*sizeof(page_t *));
#endif

    /* Initialize the tables for lower (<= 1024K) RAM. */
    for (c = 0; c < 256; c++) {
	readlookup[c] = 0xffffffff;
	writelookup[c] = 0xffffffff;
    }

    /* Initialize the tables for high (> 1024K) RAM. */
#if DYNAMIC_TABLES
    memset(readlookup2, 0xff, pages_sz*sizeof(uintptr_t));
    memset(writelookup2, 0xff, pages_sz*sizeof(uintptr_t));
#else
    memset(readlookup2, 0xff, (1<<20)*sizeof(uintptr_t));
    memset(writelookup2, 0xff, (1<<20)*sizeof(uintptr_t));
#endif

    readlnext = 0;
    writelnext = 0;
    pccache = 0xffffffff;
}


void
flushmmucache(void)
{
    int c;

    for (c = 0; c < 256; c++) {
	if (readlookup[c] != (int) 0xffffffff) {
		readlookup2[readlookup[c]] = -1;
		readlookup[c] = 0xffffffff;
	}
	if (writelookup[c] != (int) 0xffffffff) {
		page_lookup[writelookup[c]] = NULL;
		writelookup2[writelookup[c]] = -1;
		writelookup[c] = 0xffffffff;
	}
    }
    mmuflush++;

    pccache = (uint32_t)0xffffffff;
    pccache2 = (uint8_t *)0xffffffff;

#ifdef USE_DYNAREC
    codegen_flush();
#endif
}


void
flushmmucache_nopc(void)
{
    int c;

    for (c = 0; c < 256; c++) {
	if (readlookup[c] != (int) 0xffffffff) {
		readlookup2[readlookup[c]] = -1;
		readlookup[c] = 0xffffffff;
	}
	if (writelookup[c] != (int) 0xffffffff) {
		page_lookup[writelookup[c]] = NULL;
		writelookup2[writelookup[c]] = -1;
		writelookup[c] = 0xffffffff;
	}
    }
}


void
flushmmucache_cr3(void)
{
    int c;

    for (c = 0; c < 256; c++) {
	if (readlookup[c] != (int) 0xffffffff) {
		readlookup2[readlookup[c]] = -1;
		readlookup[c] = 0xffffffff;
	}
	if (writelookup[c] != (int) 0xffffffff) {
		page_lookup[writelookup[c]] = NULL;
		writelookup2[writelookup[c]] = -1;
		writelookup[c] = 0xffffffff;
	}
    }
}


void
mem_flush_write_page(uint32_t addr, uint32_t virt)
{
    page_t *page_target = &pages[addr >> 12];
    int c;

    for (c = 0; c < 256; c++) {
	if (writelookup[c] != (int) 0xffffffff) {
		uintptr_t target = (uintptr_t)&ram[(uintptr_t)(addr & ~0xfff) - (virt & ~0xfff)];

		if (writelookup2[writelookup[c]] == target || page_lookup[writelookup[c]] == page_target) {
			writelookup2[writelookup[c]] = -1;
			page_lookup[writelookup[c]] = NULL;
			writelookup[c] = 0xffffffff;
		}
	}
    }
}


#define mmutranslate_read(addr) mmutranslatereal(addr,0)
#define mmutranslate_write(addr) mmutranslatereal(addr,1)
#define rammap(x)	((uint32_t *)(_mem_exec[(x) >> MEM_GRANULARITY_BITS]))[((x) >> 2) & MEM_GRANULARITY_QMASK]
#define rammap64(x)	((uint64_t *)(_mem_exec[(x) >> MEM_GRANULARITY_BITS]))[((x) >> 3) & MEM_GRANULARITY_PMASK]

static uint64_t
mmutranslatereal_normal(uint32_t addr, int rw)
{
    uint32_t temp,temp2,temp3;
    uint32_t addr2;

    if (cpu_state.abrt)
	return 0xffffffffffffffffULL;

    addr2 = ((cr3 & ~0xfff) + ((addr >> 20) & 0xffc));
    temp = temp2 = rammap(addr2);
    if (!(temp & 1)) {
	cr2 = addr;
	temp &= 1;
	if (CPL == 3) temp |= 4;
	if (rw) temp |= 2;
	cpu_state.abrt = ABRT_PF;
	abrt_error = temp;
	return 0xffffffffffffffffULL;
    }

    if ((temp & 0x80) && (cr4 & CR4_PSE)) {
	/*4MB page*/
	if (((CPL == 3) && !(temp & 4) && !cpl_override) || (rw && !(temp & 2) && (((CPL == 3) && !cpl_override) || (cr0 & WP_FLAG)))) {
		cr2 = addr;
		temp &= 1;
		if (CPL == 3)
			temp |= 4;
		if (rw)
			temp |= 2;
		cpu_state.abrt = ABRT_PF;
		abrt_error = temp;

		return 0xffffffffffffffffULL;
	}

	mmu_perm = temp & 4;
	rammap(addr2) |= 0x20;

	return (temp & ~0x3fffff) + (addr & 0x3fffff);
    }

    temp = rammap((temp & ~0xfff) + ((addr >> 10) & 0xffc));
    temp3 = temp & temp2;
    if (!(temp&1) || ((CPL == 3) && !(temp3 & 4) && !cpl_override) || (rw && !(temp3 & 2) && (((CPL == 3) && !cpl_override) || (cr0 & WP_FLAG)))) {
	cr2 = addr;
	temp &= 1;
	if (CPL == 3) temp |= 4;
	if (rw) temp |= 2;
	cpu_state.abrt = ABRT_PF;
	abrt_error = temp;
	return 0xffffffffffffffffULL;
    }

    mmu_perm = temp & 4;
    rammap(addr2) |= 0x20;
    rammap((temp2 & ~0xfff) + ((addr >> 10) & 0xffc)) |= (rw?0x60:0x20);

    return (uint64_t) ((temp&~0xfff)+(addr&0xfff));
}


static uint64_t
mmutranslatereal_pae(uint32_t addr, int rw)
{
    uint64_t temp,temp2,temp3,temp4;
    uint64_t addr2,addr3,addr4;

    if (cpu_state.abrt)
	return 0xffffffffffffffffULL;

    addr2 = (cr3 & ~0x1f) + ((addr >> 27) & 0x18);
    temp = temp2 = rammap64(addr2) & 0x000000ffffffffffULL;
    if (!(temp & 1)) {
	cr2 = addr;
	temp &= 1;
	if (CPL == 3) temp |= 4;
	if (rw) temp |= 2;
	cpu_state.abrt = ABRT_PF;
	abrt_error = temp;
	return 0xffffffffffffffffULL;
    }

    addr3 = (temp & ~0xfffULL) + ((addr >> 18) & 0xff8);
    temp = temp4 = rammap64(addr3) & 0x000000ffffffffffULL;
    temp3 = temp & temp2;
    if (!(temp & 1)) {
	cr2 = addr;
	temp &= 1;
	if (CPL == 3) temp |= 4;
	if (rw) temp |= 2;
	cpu_state.abrt = ABRT_PF;
	abrt_error = temp;
	return 0xffffffffffffffffULL;
    }

    if (temp & 0x80) {
	/*2MB page*/
	if (((CPL == 3) && !(temp & 4) && !cpl_override) || (rw && !(temp & 2) && (((CPL == 3) && !cpl_override) || (cr0 & WP_FLAG)))) {
		cr2 = addr;
		temp &= 1;
		if (CPL == 3)
			temp |= 4;
		if (rw)
			temp |= 2;
		cpu_state.abrt = ABRT_PF;
		abrt_error = temp;

		return 0xffffffffffffffffULL;
	}
	mmu_perm = temp & 4;
	rammap64(addr3) |= 0x20;

	return ((temp & ~0x1fffffULL) + (addr & 0x1fffffULL)) & 0x000000ffffffffffULL;
    }

    addr4 = (temp & ~0xfffULL) + ((addr >> 9) & 0xff8);
    temp = rammap64(addr4) & 0x000000ffffffffffULL;
    temp3 = temp & temp4;
    if (!(temp & 1) || ((CPL == 3) && !(temp3 & 4) && !cpl_override) || (rw && !(temp3 & 2) && (((CPL == 3) && !cpl_override) || (cr0 & WP_FLAG)))) {
	cr2 = addr;
	temp &= 1;
	if (CPL == 3) temp |= 4;
	if (rw) temp |= 2;
	cpu_state.abrt = ABRT_PF;
	abrt_error = temp;
	return 0xffffffffffffffffULL;
    }

    mmu_perm = temp & 4;
    rammap64(addr3) |= 0x20;
    rammap64(addr4) |= (rw? 0x60 : 0x20);

    return ((temp & ~0xfffULL) + ((uint64_t) (addr & 0xfff))) & 0x000000ffffffffffULL;
}


uint64_t
mmutranslatereal(uint32_t addr, int rw)
{
    if (cr4 & CR4_PAE)
	return mmutranslatereal_pae(addr, rw);
    else
	return mmutranslatereal_normal(addr, rw);
}


/* This is needed because the old recompiler calls this to check for page fault. */
uint32_t
mmutranslatereal32(uint32_t addr, int rw)
{
    return (uint32_t) mmutranslatereal(addr, rw);
}


static uint64_t
mmutranslate_noabrt_normal(uint32_t addr, int rw)
{
    uint32_t temp,temp2,temp3;
    uint32_t addr2;

    if (cpu_state.abrt) 
	return 0xffffffffffffffffULL;

    addr2 = ((cr3 & ~0xfff) + ((addr >> 20) & 0xffc));
    temp = temp2 = rammap(addr2);

    if (! (temp & 1))
	return 0xffffffffffffffffULL;

    if ((temp & 0x80) && (cr4 & CR4_PSE)) {
	/*4MB page*/
	if (((CPL == 3) && !(temp & 4) && !cpl_override) || (rw && !(temp & 2) && ((CPL == 3) || (cr0 & WP_FLAG))))
		return 0xffffffffffffffffULL;

	return (temp & ~0x3fffff) + (addr & 0x3fffff);
    }

    temp = rammap((temp & ~0xfff) + ((addr >> 10) & 0xffc));
    temp3 = temp & temp2;

    if (!(temp & 1) || ((CPL == 3) && !(temp3 & 4) && !cpl_override) || (rw && !(temp3 & 2) && ((CPL == 3) || (cr0 & WP_FLAG))))
	return 0xffffffffffffffffULL;

    return (uint64_t) ((temp & ~0xfff) + (addr & 0xfff));
}


static uint64_t
mmutranslate_noabrt_pae(uint32_t addr, int rw)
{
    uint64_t temp,temp2,temp3,temp4;
    uint64_t addr2,addr3,addr4;

    if (cpu_state.abrt) 
	return 0xffffffffffffffffULL;

    addr2 = (cr3 & ~0x1f) + ((addr >> 27) & 0x18);
    temp = temp2 = rammap64(addr2) & 0x000000ffffffffffULL;

    if (! (temp & 1))
	return 0xffffffffffffffffULL;

    addr3 = (temp & ~0xfffULL) + ((addr >> 18) & 0xff8);
    temp = temp4 = rammap64(addr3) & 0x000000ffffffffffULL;
    temp3 = temp & temp2;

    if (! (temp & 1))
	return 0xffffffffffffffffULL;

    if (temp & 0x80) {
	/*2MB page*/
	if (((CPL == 3) && !(temp & 4) && !cpl_override) || (rw && !(temp & 2) && ((CPL == 3) || (cr0 & WP_FLAG))))
		return 0xffffffffffffffffULL;

	return ((temp & ~0x1fffffULL) + (addr & 0x1fffff)) & 0x000000ffffffffffULL;
    }

    addr4 = (temp & ~0xfffULL) + ((addr >> 9) & 0xff8);
    temp = rammap64(addr4) & 0x000000ffffffffffULL;;
    temp3 = temp & temp4;

    if (!(temp&1) || ((CPL == 3) && !(temp3 & 4) && !cpl_override) || (rw && !(temp3 & 2) && ((CPL == 3) || (cr0 & WP_FLAG))))
	return 0xffffffffffffffffULL;

    return ((temp & ~0xfffULL) + ((uint64_t) (addr & 0xfff))) & 0x000000ffffffffffULL;
}


uint64_t
mmutranslate_noabrt(uint32_t addr, int rw)
{
    if (cr4 & CR4_PAE)
	return mmutranslate_noabrt_pae(addr, rw);
    else
	return mmutranslate_noabrt_normal(addr, rw);
}


void
mmu_invalidate(uint32_t addr)
{
    flushmmucache_cr3();
}


uint8_t
mem_addr_range_match(uint32_t addr, uint32_t start, uint32_t len)
{
    if (addr < start)
	return 0;
      else if (addr >= (start + len))
	return 0;
      else
	return 1;
}


uint32_t
mem_addr_translate(uint32_t addr, uint32_t chunk_start, uint32_t len)
{
    uint32_t mask = len - 1;

    return chunk_start + (addr & mask);
}


void
addreadlookup(uint32_t virt, uint32_t phys)
{
    if (virt == 0xffffffff) return;

    if (readlookup2[virt>>12] != (uintptr_t) -1) return;

    if (readlookup[readlnext] != (int) 0xffffffff)
	readlookup2[readlookup[readlnext]] = -1;

    readlookup2[virt>>12] = (uintptr_t)&ram[(uintptr_t)(phys & ~0xFFF) - (uintptr_t)(virt & ~0xfff)];

    readlookupp[readlnext] = mmu_perm;
    readlookup[readlnext++] = virt >> 12;
    readlnext &= (cachesize-1);

    sub_cycles(9);
}


void
addwritelookup(uint32_t virt, uint32_t phys)
{
    if (virt == 0xffffffff) return;

    if (page_lookup[virt >> 12]) return;

    if (writelookup[writelnext] != -1) {
	page_lookup[writelookup[writelnext]] = NULL;
	writelookup2[writelookup[writelnext]] = -1;
    }

#ifdef USE_NEW_DYNAREC
    if (pages[phys >> 12].block || (phys & ~0xfff) == recomp_page)
#else
#ifdef USE_DYNAREC
    if (pages[phys >> 12].block[0] || pages[phys >> 12].block[1] || pages[phys >> 12].block[2] || pages[phys >> 12].block[3] || (phys & ~0xfff) == recomp_page)
#else
    if (pages[phys >> 12].block[0] || pages[phys >> 12].block[1] || pages[phys >> 12].block[2] || pages[phys >> 12].block[3])
#endif
#endif
	page_lookup[virt >> 12] = &pages[phys >> 12];
    else
	writelookup2[virt>>12] = (uintptr_t)&ram[(uintptr_t)(phys & ~0xFFF) - (uintptr_t)(virt & ~0xfff)];

    writelookupp[writelnext] = mmu_perm;
    writelookup[writelnext++] = virt >> 12;
    writelnext &= (cachesize - 1);

    sub_cycles(9);
}


uint8_t *
getpccache(uint32_t a)
{
    uint64_t a64 = (uint64_t) a;
    uint32_t a2;

    a2 = a;

    if (cr0 >> 31) {
	a64 = mmutranslate_read(a64);

	if (a64 == 0xffffffffffffffffULL) return ram;
    }
    a64 &= rammask;

    if (_mem_exec[a64 >> MEM_GRANULARITY_BITS]) {
	if (is286) {
		if (read_mapping[a64 >> MEM_GRANULARITY_BITS] && (read_mapping[a64 >> MEM_GRANULARITY_BITS]->flags & MEM_MAPPING_ROM))
			cpu_prefetch_cycles = cpu_rom_prefetch_cycles;
		else
			cpu_prefetch_cycles = cpu_mem_prefetch_cycles;
	}
	
	return &_mem_exec[a64 >> MEM_GRANULARITY_BITS][(uintptr_t)(a64 & MEM_GRANULARITY_PAGE) - (uintptr_t)(a2 & ~0xfff)];
    }

    mem_log("Bad getpccache %08X%08X\n", (uint32_t) (a >> 32), (uint32_t) (a & 0xffffffff));

#if FIXME
    return &ff_array[0-(uintptr_t)(a2 & ~0xfff)];
#else
    return (uint8_t *)&ff_pccache;
#endif
}


uint8_t
readmembl(uint32_t addr)
{
    uint64_t addr64 = (uint64_t) addr;
    mem_mapping_t *map;

    mem_logical_addr = addr;

    if (cr0 >> 31) {
	addr64 = mmutranslate_read(addr);
	if (addr64 == 0xffffffffffffffffULL)
		return 0xff;
	if (addr64 > 0xffffffffULL)
		return 0xff;
    }
    addr = (uint32_t) (addr64 & rammask);

    map = read_mapping[addr >> MEM_GRANULARITY_BITS];
    if (map && map->read_b)
	return map->read_b(addr, map->p);

    return 0xff;
}


void
writemembl(uint32_t addr, uint8_t val)
{
    uint64_t addr64 = (uint64_t) addr;
    mem_mapping_t *map;
    mem_logical_addr = addr;

    if (page_lookup[addr>>12]) {
	page_lookup[addr>>12]->write_b(addr, val, page_lookup[addr>>12]);
	return;
    }

    if (cr0 >> 31) {
	addr64 = mmutranslate_write(addr);
	if (addr64 == 0xffffffffffffffffULL)
		return;
	if (addr64 > 0xffffffffULL)
		return;
    }
    addr = (uint32_t) (addr64 & rammask);

    map = write_mapping[addr >> MEM_GRANULARITY_BITS];
    if (map && map->write_b)
	map->write_b(addr, val, map->p);
}


#ifdef USE_NEW_DYNAREC
uint16_t
readmemwl(uint32_t addr)
{
    uint64_t addr64 = (uint64_t) addr;
    mem_mapping_t *map;

    mem_logical_addr = addr;

    if (addr64 & 1) {
	if (!cpu_cyrix_alignment || (addr64 & 7) == 7)
		sub_cycles(timing_misaligned);
	if ((addr64 & 0xfff) > 0xffe) {
		if (cr0 >> 31) {
			if (mmutranslate_read(addr)   == 0xffffffffffffffffULL)
				return 0xffff;
			if (mmutranslate_read(addr+1) == 0xffffffffffffffffULL)
				return 0xffff;
		}
		return readmembl(addr)|(readmembl(addr+1)<<8);
	} else if (readlookup2[addr >> 12] != -1)
		return *(uint16_t *)(readlookup2[addr >> 12] + addr);
    }
    if (cr0>>31) {
	addr64 = mmutranslate_read(addr);
	if (addr64 == 0xffffffffffffffffULL)
		return 0xffff;
	if (addr64 > 0xffffffffULL)
		return 0xffff;
    }

    addr = (uint32_t) (addr64 & rammask);

    map = read_mapping[addr >> MEM_GRANULARITY_BITS];

    if (map && map->read_w)
	return map->read_w(addr, map->p);

    if (map && map->read_b)
	return map->read_b(addr, map->p) | (map->read_b(addr + 1, map->p) << 8);

    return 0xffff;
}


void
writememwl(uint32_t addr, uint16_t val)
{
    uint64_t addr64 = (uint64_t) addr;
    mem_mapping_t *map;

    mem_logical_addr = addr;

    if (addr & 1) {
	if (!cpu_cyrix_alignment || (addr & 7) == 7)
		sub_cycles(timing_misaligned);
	if ((addr & 0xFFF) > 0xFFE) {
		if (cr0 >> 31) {
			if (mmutranslate_write(addr)   == 0xffffffff)
				return;
			if (mmutranslate_write(addr+1) == 0xffffffff)
				return;
		}
		writemembl(addr,val);
		writemembl(addr+1,val>>8);
		return;
	} else if (writelookup2[addr >> 12] != -1) {
		*(uint16_t *)(writelookup2[addr >> 12] + addr) = val;
		return;
	}
    }

    if (page_lookup[addr>>12]) {
	page_lookup[addr>>12]->write_w(addr, val, page_lookup[addr>>12]);
	return;
    }
    if (cr0>>31) {
	addr64 = mmutranslate_write(addr);
	if (addr64 == 0xffffffffffffffffULL)
		return;
	if (addr64 > 0xffffffffULL)
		return;
    }

    addr = (uint32_t) (addr64 & rammask);

    map = write_mapping[addr >> MEM_GRANULARITY_BITS];
    if (map) {
	if (map->write_w)
		map->write_w(addr, val, map->p);
	else if (map->write_b) {
		map->write_b(addr, val, map->p);
		map->write_b(addr + 1, val >> 8, map->p);
	}
    }
}


uint32_t
readmemll(uint32_t addr)
{
    uint64_t addr64 = (uint64_t) addr;
    mem_mapping_t *map;

    mem_logical_addr = addr;

    if (addr & 3) {
	if (!cpu_cyrix_alignment || (addr & 7) > 4)
		sub_cycles(timing_misaligned);
	if ((addr & 0xfff) > 0xffc) {
		if (cr0>>31) {
			if (mmutranslate_read(addr)   == 0xffffffffffffffffULL)
				return 0xffffffff;
			if (mmutranslate_read(addr+3) == 0xffffffffffffffffULL)
				return 0xffffffff;
		}
		return readmemwl(addr)|(readmemwl(addr+2)<<16);
	} else if (readlookup2[addr >> 12] != -1)
		return *(uint32_t *)(readlookup2[addr >> 12] + addr);
    }

    if (cr0 >> 31) {
	addr64 = mmutranslate_read(addr);
	if (addr64 == 0xffffffffffffffffULL)
		return 0xffffffff;
	if (addr64 > 0xffffffffULL)
		return 0xffffffff;
    }

    addr = (uint32_t) (addr64 & rammask);

    map = read_mapping[addr >> MEM_GRANULARITY_BITS];
    if (map) {
	if (map->read_l)
		return map->read_l(addr, map->p);

	if (map->read_w)
		return map->read_w(addr, map->p) | (map->read_w(addr + 2, map->p) << 16);

	if (map->read_b)
		return map->read_b(addr, map->p) | (map->read_b(addr + 1, map->p) << 8) |
		       (map->read_b(addr + 2, map->p) << 16) | (map->read_b(addr + 3, map->p) << 24);
    }

    return 0xffffffff;
}


void
writememll(uint32_t addr, uint32_t val)
{
    uint64_t addr64 = (uint64_t) addr;
    mem_mapping_t *map;

    mem_logical_addr = addr;

    if (addr & 3) {
	if (!cpu_cyrix_alignment || (addr & 7) > 4)
		sub_cycles(timing_misaligned);
	if ((addr & 0xFFF) > 0xFFC) {
		if (cr0>>31) {
			if (mmutranslate_write(addr)   == 0xffffffffffffffffULL)
				return;
			if (mmutranslate_write(addr+3) == 0xffffffffffffffffULL)
				return;
		}
		writememwl(addr,val);
		writememwl(addr+2,val>>16);
		return;
	} else if (writelookup2[addr >> 12] != -1) {
		*(uint32_t *)(writelookup2[addr >> 12] + addr) = val;
		return;
	}
    }
    if (page_lookup[addr>>12]) {
	page_lookup[addr>>12]->write_l(addr, val, page_lookup[addr>>12]);
	return;
    }
    if (cr0>>31) {
	addr64 = mmutranslate_write(addr);
	if (addr64 == 0xffffffffffffffffULL)
		return;
	if (addr64 > 0xffffffffULL)
		return;
    }

    addr = (uint32_t) (addr64 & rammask);

    map = write_mapping[addr >> MEM_GRANULARITY_BITS];
    if (map) {
	if (map->write_l)
		map->write_l(addr, val, map->p);
	else if (map->write_w) {
		map->write_w(addr, val, map->p);
		map->write_w(addr + 2, val >> 16, map->p);
	} else if (map->write_b) {
		map->write_b(addr, val, map->p);
		map->write_b(addr + 1, val >> 8, map->p);
		map->write_b(addr + 2, val >> 16, map->p);
		map->write_b(addr + 3, val >> 24, map->p);
	}
    }
}


uint64_t
readmemql(uint32_t addr)
{
    uint64_t addr64 = (uint64_t) addr;
    mem_mapping_t *map;

    mem_logical_addr = addr;

    if (addr & 7) {
	sub_cycles(timing_misaligned);
	if ((addr & 0xFFF) > 0xFF8) {
		if (cr0>>31) {
			if (mmutranslate_read(addr)   == 0xffffffffffffffffULL)
				return 0xffffffffffffffffULL;
			if (mmutranslate_read(addr+7) == 0xffffffffffffffffULL)
				return 0xffffffffffffffffULL;
		}
		return readmemll(addr)|((uint64_t)readmemll(addr+4)<<32);
	} else if (readlookup2[addr >> 12] != -1)
		return *(uint64_t *)(readlookup2[addr >> 12] + addr);
    }

    if (cr0>>31) {
	addr64 = mmutranslate_read(addr);
	if (addr64 == 0xffffffffffffffffULL)
		return 0xffffffffffffffffULL;
	if (addr64 > 0xffffffffULL)
		return 0xffffffffffffffffULL;
    }

    addr  = (uint32_t) (addr64 & rammask);

    map = read_mapping[addr >> MEM_GRANULARITY_BITS];
    if (map && map->read_l)
	return map->read_l(addr, map->p) | ((uint64_t)map->read_l(addr + 4, map->p) << 32);

    return readmemll(addr) | ((uint64_t)readmemll(addr+4)<<32);
}


void
writememql(uint32_t addr, uint64_t val)
{
    uint64_t addr64 = (uint64_t) addr;
    mem_mapping_t *map;

    mem_logical_addr = addr;

    if (addr & 7) {
	sub_cycles(timing_misaligned);
	if ((addr & 0xFFF) > 0xFF8) {
		if (cr0>>31) {
			if (mmutranslate_write(addr)   == 0xffffffffffffffffULL)
				return;
			if (mmutranslate_write(addr+7) == 0xffffffffffffffffULL)
				return;
		}
		writememll(addr, val);
		writememll(addr+4, val >> 32);
		return;
	} else if (writelookup2[addr >> 12] != -1) {
		*(uint64_t *)(writelookup2[addr >> 12] + addr) = val;
		return;
	}
    }
    if (page_lookup[addr>>12]) {
	page_lookup[addr>>12]->write_l(addr, val, page_lookup[addr>>12]);
	page_lookup[addr>>12]->write_l(addr + 4, val >> 32, page_lookup[addr>>12]);
	return;
    }
    if (cr0>>31) {
	addr64 = mmutranslate_write(addr);
	if (addr64 == 0xffffffffffffffffULL)
		return;
	if (addr64 > 0xffffffffULL)
		return;
    }

    addr = (uint32_t) (addr64 & rammask);

    map = write_mapping[addr >> MEM_GRANULARITY_BITS];
    if (map) {
	if (map->write_l) {
		map->write_l(addr, val, map->p);
		map->write_l(addr + 4, val >> 32, map->p);
	} else if (map->write_w) {
		map->write_w(addr, val, map->p);
		map->write_w(addr + 2, val >> 16, map->p);
		map->write_w(addr + 4, val >> 32, map->p);
		map->write_w(addr + 6, val >> 48, map->p);
	} else if (map->write_b) {
		map->write_b(addr, val, map->p);
		map->write_b(addr + 1, val >> 8, map->p);
		map->write_b(addr + 2, val >> 16, map->p);
		map->write_b(addr + 3, val >> 24, map->p);
		map->write_b(addr + 4, val >> 32, map->p);
		map->write_b(addr + 5, val >> 40, map->p);
		map->write_b(addr + 6, val >> 48, map->p);
		map->write_b(addr + 7, val >> 56, map->p);
	}
    }
}
#else
uint8_t
readmemb386l(uint32_t seg, uint32_t addr)
{
    return readmembl(addr + seg);
}


void
writememb386l(uint32_t seg, uint32_t addr, uint8_t val)
{
    writemembl(addr + seg, val);
}


uint16_t
readmemwl(uint32_t seg, uint32_t addr)
{
    uint64_t addr64 = (uint64_t) addr;
    mem_mapping_t *map;
    uint32_t addr2 = mem_logical_addr = seg + addr;

    if (addr2 & 1) {
	if (!cpu_cyrix_alignment || (addr2 & 7) == 7)
		sub_cycles(timing_misaligned);
	if ((addr2 & 0xfff) > 0xffe) {
		if (cr0 >> 31) {
			if (mmutranslate_read(addr2)   == 0xffffffffffffffffULL)
				return 0xffff;
			if (mmutranslate_read(addr2+1) == 0xffffffffffffffffULL)
				return 0xffff;
		}
		if (is386) return readmemb386l(seg,addr)|(((uint16_t) readmemb386l(seg,addr+1))<<8);
		else       return readmembl(seg+addr)|(((uint16_t) readmembl(seg+addr+1))<<8);
	}
	else if (readlookup2[addr2 >> 12] != (uintptr_t) -1)
		return *(uint16_t *)(readlookup2[addr2 >> 12] + addr2);
    }

    if (cr0 >> 31) {
	addr64 = mmutranslate_read(addr2);
	if (addr64 == 0xffffffffffffffffULL)
		return 0xffff;
	if (addr64 > 0xffffffffULL)
		return 0xffff;
    } else
	addr64 = (uint64_t) addr2;

    addr2 = (uint32_t) (addr64 & rammask);

    map = read_mapping[addr2 >> MEM_GRANULARITY_BITS];

    if (map && map->read_w)
	return map->read_w(addr2, map->p);

    if (map && map->read_b) {
	if (AT)
		return map->read_b(addr2, map->p) |
		       ((uint16_t) (map->read_b(addr2 + 1, map->p)) << 8);
	else
		return map->read_b(addr2, map->p) |
		       ((uint16_t) (map->read_b(seg + ((addr + 1) & 0xffff), map->p)) << 8);
    }

    return 0xffff;
}


void
writememwl(uint32_t seg, uint32_t addr, uint16_t val)
{
    uint64_t addr64 = (uint64_t) addr;
    mem_mapping_t *map;
    uint32_t addr2 = mem_logical_addr = seg + addr;

    if (addr2 & 1) {
	if (!cpu_cyrix_alignment || (addr2 & 7) == 7)
		sub_cycles(timing_misaligned);
	if ((addr2 & 0xFFF) > 0xffe) {
		if (cr0 >> 31) {
			if (mmutranslate_write(addr2)   == 0xffffffffffffffffULL) return;
			if (mmutranslate_write(addr2+1) == 0xffffffffffffffffULL) return;
		}
		if (is386) {
			writememb386l(seg,addr,val);
			writememb386l(seg,addr+1,val>>8);
		} else {
			writemembl(seg+addr,val);
			writemembl(seg+addr+1,val>>8);
		}
		return;
	} else if (writelookup2[addr2 >> 12] != (uintptr_t) -1) {
		*(uint16_t *)(writelookup2[addr2 >> 12] + addr2) = val;
		return;
	}
    }

    if (page_lookup[addr2>>12]) {
	page_lookup[addr2>>12]->write_w(addr2, val, page_lookup[addr2>>12]);
	return;
    }

    if (cr0 >> 31) {
	addr64 = mmutranslate_write(addr2);
	if (addr64 == 0xffffffffffffffffULL)
		return;
	if (addr64 > 0xffffffffULL)
		return;
    } else
	addr64 = (uint64_t) addr2;

    addr2 = (uint32_t) (addr64 & rammask);

    map = write_mapping[addr2 >> MEM_GRANULARITY_BITS];

    if (map && map->write_w) {
	map->write_w(addr2, val, map->p);
	return;
    }

    if (map && map->write_b) {
	map->write_b(addr2, val, map->p);
	map->write_b(addr2 + 1, val >> 8, map->p);
	return;
    }
}


uint32_t
readmemll(uint32_t seg, uint32_t addr)
{
    uint64_t addr64 = (uint64_t) addr;
    mem_mapping_t *map;
    uint32_t addr2 = mem_logical_addr = seg + addr;

    if (addr2 & 3) {
	if (!cpu_cyrix_alignment || (addr2 & 7) > 4)
		sub_cycles(timing_misaligned);
	if ((addr2 & 0xfff) > 0xffc) {
		if (cr0 >> 31) {
			if (mmutranslate_read(addr2)   == 0xffffffffffffffffULL) return 0xffffffff;
			if (mmutranslate_read(addr2+3) == 0xffffffffffffffffULL) return 0xffffffff;
		}
		return readmemwl(seg,addr)|(readmemwl(seg,addr+2)<<16);
	} else if (readlookup2[addr2 >> 12] != (uintptr_t) -1)
		return *(uint32_t *)(readlookup2[addr2 >> 12] + addr2);
    }

    if (cr0 >> 31) {
	addr64 = mmutranslate_read(addr2);
	if (addr64 == 0xffffffffffffffffULL)
		return 0xffffffff;
	if (addr64 > 0xffffffffULL)
		return 0xffffffff;
    } else
	addr64 = (uint64_t) addr2;

    addr2 = (uint32_t) (addr64 & rammask);

    map = read_mapping[addr2 >> MEM_GRANULARITY_BITS];

    if (map && map->read_l)
	return map->read_l(addr2, map->p);

    if (map && map->read_w)
	return map->read_w(addr2, map->p) |
	       ((uint32_t) (map->read_w(addr2 + 2, map->p)) << 16);

    if (map && map->read_b)
	return map->read_b(addr2, map->p) |
	       ((uint32_t) (map->read_b(addr2 + 1, map->p)) << 8) |
	       ((uint32_t) (map->read_b(addr2 + 2, map->p)) << 16) |
	       ((uint32_t) (map->read_b(addr2 + 3, map->p)) << 24);

    return 0xffffffff;
}


void
writememll(uint32_t seg, uint32_t addr, uint32_t val)
{
    uint64_t addr64 = (uint64_t) addr;
    mem_mapping_t *map;
    uint32_t addr2 = mem_logical_addr = seg + addr;

    if (addr2 & 3) {
	if (!cpu_cyrix_alignment || (addr2 & 7) > 4)
		sub_cycles(timing_misaligned);
	if ((addr2 & 0xfff) > 0xffc) {
		if (cr0 >> 31) {
			if (mmutranslate_write(addr2)   == 0xffffffffffffffffULL) return;
			if (mmutranslate_write(addr2+3) == 0xffffffffffffffffULL) return;
		}
		writememwl(seg,addr,val);
		writememwl(seg,addr+2,val>>16);
		return;
	} else if (writelookup2[addr2 >> 12] != (uintptr_t) -1) {
		*(uint32_t *)(writelookup2[addr2 >> 12] + addr2) = val;
		return;
	}
    }

    if (page_lookup[addr2>>12]) {
	page_lookup[addr2>>12]->write_l(addr2, val, page_lookup[addr2>>12]);
	return;
    }

    if (cr0 >> 31) {
	addr64 = mmutranslate_write(addr2);
	if (addr64 == 0xffffffffffffffffULL)
		return;
	if (addr64 > 0xffffffffULL)
		return;
    } else
	addr64 = (uint32_t) addr2;

    addr2 = (uint32_t) (addr64 & rammask);

    map = write_mapping[addr2 >> MEM_GRANULARITY_BITS];

    if (map && map->write_l) {
	map->write_l(addr2, val,	   map->p);
	return;
    }
    if (map && map->write_w) {
	map->write_w(addr2,     val,       map->p);
	map->write_w(addr2 + 2, val >> 16, map->p);
	return;
    }
    if (map && map->write_b) {
	map->write_b(addr2,     val,       map->p);
	map->write_b(addr2 + 1, val >> 8,  map->p);
	map->write_b(addr2 + 2, val >> 16, map->p);
	map->write_b(addr2 + 3, val >> 24, map->p);
	return;
    }
}


uint64_t
readmemql(uint32_t seg, uint32_t addr)
{
    uint64_t addr64 = (uint64_t) addr;
    mem_mapping_t *map;
    uint32_t addr2 = mem_logical_addr = seg + addr;

    if (addr2 & 7) {
	sub_cycles(timing_misaligned);
	if ((addr2 & 0xfff) > 0xff8) {
		if (cr0 >> 31) {
			if (mmutranslate_read(addr2)   == 0xffffffffffffffffULL) return 0xffffffffffffffffULL;
			if (mmutranslate_read(addr2+7) == 0xffffffffffffffffULL) return 0xffffffffffffffffULL;
		}
		return readmemll(seg,addr)|((uint64_t)readmemll(seg,addr+4)<<32);
	} else if (readlookup2[addr2 >> 12] != (uintptr_t) -1)
		return *(uint64_t *)(readlookup2[addr2 >> 12] + addr2);
    }

    if (cr0 >> 31) {
	addr64 = mmutranslate_read(addr2);
	if (addr64 == 0xffffffffffffffffULL)
		return 0xffffffffffffffffULL;
	if (addr64 > 0xffffffffULL)
		return 0xffffffffffffffffULL;
    } else
	addr64 = (uint64_t) addr2;

    addr2 = (uint32_t) (addr64 & rammask);

    map = read_mapping[addr2 >> MEM_GRANULARITY_BITS];
    if (map && map->read_l)
	return map->read_l(addr2, map->p) | ((uint64_t)map->read_l(addr2 + 4, map->p) << 32);

    return readmemll(seg,addr) | ((uint64_t)readmemll(seg,addr+4)<<32);
}


void
writememql(uint32_t seg, uint32_t addr, uint64_t val)
{
    uint64_t addr64 = (uint64_t) addr;
    mem_mapping_t *map;
    uint32_t addr2 = mem_logical_addr = seg + addr;

    if (addr2 & 7) {
	sub_cycles(timing_misaligned);
	if ((addr2 & 0xfff) > 0xff8) {
		if (cr0 >> 31) {
			if (mmutranslate_write(addr2)   == 0xffffffffffffffffULL) return;
			if (mmutranslate_write(addr2+7) == 0xffffffffffffffffULL) return;
		}
		writememll(seg, addr, val);
		writememll(seg, addr+4, val >> 32);
		return;
	} else if (writelookup2[addr2 >> 12] != (uintptr_t) -1) {
		*(uint64_t *)(writelookup2[addr2 >> 12] + addr2) = val;
		return;
	}
    }

    if (page_lookup[addr2>>12]) {
	page_lookup[addr2>>12]->write_l(addr2, val, page_lookup[addr2>>12]);
	page_lookup[addr2>>12]->write_l(addr2 + 4, val >> 32, page_lookup[addr2>>12]);
	return;
    }

    if (cr0 >> 31) {
	addr64 = mmutranslate_write(addr2);
	if (addr64 == 0xffffffffffffffffULL)
		return;
	if (addr64 > 0xffffffffULL)
		return;
    } else
	addr64 = (uint64_t) addr2;

    addr2 = (uint32_t) (addr64 & rammask);

    map = write_mapping[addr2 >> MEM_GRANULARITY_BITS];

    if (map && map->write_l) {
	map->write_l(addr2,   val,       map->p);
	map->write_l(addr2+4, val >> 32, map->p);
	return;
    }
    if (map && map->write_w) {
	map->write_w(addr2,     val,       map->p);
	map->write_w(addr2 + 2, val >> 16, map->p);
	map->write_w(addr2 + 4, val >> 32, map->p);
	map->write_w(addr2 + 6, val >> 48, map->p);
	return;
    }
    if (map && map->write_b) {
	map->write_b(addr2,     val,       map->p);
	map->write_b(addr2 + 1, val >> 8,  map->p);
	map->write_b(addr2 + 2, val >> 16, map->p);
	map->write_b(addr2 + 3, val >> 24, map->p);
	map->write_b(addr2 + 4, val >> 32, map->p);
	map->write_b(addr2 + 5, val >> 40, map->p);
	map->write_b(addr2 + 6, val >> 48, map->p);
	map->write_b(addr2 + 7, val >> 56, map->p);
	return;
    }
}
#endif


int
mem_mapping_is_romcs(uint32_t addr, int write)
{
    mem_mapping_t *map;

    if (write)
	map = write_mapping[addr >> MEM_GRANULARITY_BITS];
    else
	map = read_mapping[addr >> MEM_GRANULARITY_BITS];

    if (map)
	return !!(map->flags & MEM_MAPPING_ROMCS);
    else
	return 0;
}


uint8_t
mem_readb_phys(uint32_t addr)
{
    mem_mapping_t *map = read_mapping[addr >> MEM_GRANULARITY_BITS];

    mem_logical_addr = 0xffffffff;

#ifdef USE_PHYS_EXEC
    if (_mem_exec[addr >> MEM_GRANULARITY_BITS])
	return _mem_exec[addr >> MEM_GRANULARITY_BITS][addr & MEM_GRANULARITY_MASK];
    else
#endif
    if (map && map->read_b)
       	return map->read_b(addr, map->p);
    else
	return 0xff;
}


uint16_t
mem_readw_phys(uint32_t addr)
{
    mem_mapping_t *map = read_mapping[addr >> MEM_GRANULARITY_BITS];
    uint16_t temp;
#ifdef USE_PHYS_EXEC
    uint16_t *p;
#endif

    mem_logical_addr = 0xffffffff;

#ifdef USE_PHYS_EXEC
    if ((addr <= MEM_GRANULARITY_HBOUND) && (_mem_exec[addr >> MEM_GRANULARITY_BITS])) {
	p = (uint16_t *) &(_mem_exec[addr >> MEM_GRANULARITY_BITS][addr & MEM_GRANULARITY_MASK]);
	return *p;
    } else
#endif
    if ((addr <= MEM_GRANULARITY_HBOUND) && (map && map->read_w))
       	return map->read_w(addr, map->p);
    else {
	temp = mem_readb_phys(addr + 1) << 8;
	temp |=  mem_readb_phys(addr);
    }

    return temp;
}


uint32_t
mem_readl_phys(uint32_t addr)
{
    mem_mapping_t *map = read_mapping[addr >> MEM_GRANULARITY_BITS];
    uint32_t temp;
#ifdef USE_PHYS_EXEC
    uint32_t *p;
#endif

    mem_logical_addr = 0xffffffff;

#ifdef USE_PHYS_EXEC
    if ((addr <= MEM_GRANULARITY_QBOUND) && (_mem_exec[addr >> MEM_GRANULARITY_BITS])) {
	p = (uint32_t *) &(_mem_exec[addr >> MEM_GRANULARITY_BITS][addr & MEM_GRANULARITY_MASK]);
	return *p;
    } else
#endif
    if ((addr <= MEM_GRANULARITY_QBOUND) && (map && map->read_l))
       	return map->read_l(addr, map->p);
    else {
	temp = mem_readw_phys(addr + 2) << 16;
	temp |=  mem_readw_phys(addr);
    }

    return temp;
}


void
mem_read_phys(void *dest, uint32_t addr, int transfer_size)
{
    uint8_t *pb;
    uint16_t *pw;
    uint32_t *pl;

    if (transfer_size == 4) {
	pl = (uint32_t *) dest;
	*pl = mem_readl_phys(addr);
    } else if (transfer_size == 2) {
	pw = (uint16_t *) dest;
	*pw = mem_readw_phys(addr);
    } else if (transfer_size == 4) {
	pb = (uint8_t *) dest;
	*pb = mem_readb_phys(addr);
    }
}


void
mem_writeb_phys(uint32_t addr, uint8_t val)
{
    mem_mapping_t *map = write_mapping[addr >> MEM_GRANULARITY_BITS];

    mem_logical_addr = 0xffffffff;

#ifdef USE_PHYS_EXEC
    if (_mem_exec[addr >> MEM_GRANULARITY_BITS])
	_mem_exec[addr >> MEM_GRANULARITY_BITS][addr & MEM_GRANULARITY_MASK] = val;
    else
#endif
    if (map && map->write_b)
       	map->write_b(addr, val, map->p);
}


void
mem_writew_phys(uint32_t addr, uint16_t val)
{
    mem_mapping_t *map = write_mapping[addr >> MEM_GRANULARITY_BITS];
#ifdef USE_PHYS_EXEC
    uint16_t *p;
#endif

    mem_logical_addr = 0xffffffff;

#ifdef USE_PHYS_EXEC
    if ((addr <= MEM_GRANULARITY_HBOUND) && (_mem_exec[addr >> MEM_GRANULARITY_BITS])) {
	p = (uint16_t *) &(_mem_exec[addr >> MEM_GRANULARITY_BITS][addr & MEM_GRANULARITY_MASK]);
	*p = val;
    } else
#endif
    if ((addr <= MEM_GRANULARITY_HBOUND) && (map && map->write_w))
       	map->write_w(addr, val, map->p);
    else {
	mem_writeb_phys(addr, val & 0xff);
	mem_writeb_phys(addr + 1, (val >> 8) & 0xff);
    }
}


void
mem_writel_phys(uint32_t addr, uint32_t val)
{
    mem_mapping_t *map = write_mapping[addr >> MEM_GRANULARITY_BITS];
#ifdef USE_PHYS_EXEC
    uint32_t *p;
#endif

    mem_logical_addr = 0xffffffff;

#ifdef USE_PHYS_EXEC
    if ((addr <= MEM_GRANULARITY_QBOUND) && (_mem_exec[addr >> MEM_GRANULARITY_BITS])) {
	p = (uint32_t *) &(_mem_exec[addr >> MEM_GRANULARITY_BITS][addr & MEM_GRANULARITY_MASK]);
	*p = val;
    } else
#endif
    if ((addr <= MEM_GRANULARITY_QBOUND) && (map && map->write_l))
       	map->write_l(addr, val, map->p);
    else {
	mem_writew_phys(addr, val & 0xffff);
	mem_writew_phys(addr + 2, (val >> 16) & 0xffff);
    }
}


void
mem_write_phys(void *src, uint32_t addr, int transfer_size)
{
    uint8_t *pb;
    uint16_t *pw;
    uint32_t *pl;

    if (transfer_size == 4) {
	pl = (uint32_t *) src;
	mem_writel_phys(addr, *pl);
    } else if (transfer_size == 2) {
	pw = (uint16_t *) src;
	mem_writew_phys(addr, *pw);
    } else {
	pb = (uint8_t *) src;
	mem_writeb_phys(addr, *pb);
    }
}


uint8_t
mem_read_ram(uint32_t addr, void *priv)
{
#ifdef ENABLE_MEM_LOG
    if ((addr >= 0xa0000) && (addr <= 0xbffff))
	mem_log("Read  B       %02X from %08X\n", ram[addr], addr);
#endif

    addreadlookup(mem_logical_addr, addr);

    return ram[addr];
}


uint16_t
mem_read_ramw(uint32_t addr, void *priv)
{
#ifdef ENABLE_MEM_LOG
    if ((addr >= 0xa0000) && (addr <= 0xbffff))
	mem_log("Read  W     %04X from %08X\n", *(uint16_t *)&ram[addr], addr);
#endif

    addreadlookup(mem_logical_addr, addr);

    return *(uint16_t *)&ram[addr];
}


uint32_t
mem_read_raml(uint32_t addr, void *priv)
{
#ifdef ENABLE_MEM_LOG
    if ((addr >= 0xa0000) && (addr <= 0xbffff))
	mem_log("Read  L %08X from %08X\n", *(uint32_t *)&ram[addr], addr);
#endif

    addreadlookup(mem_logical_addr, addr);

    return *(uint32_t *)&ram[addr];
}


#ifdef USE_NEW_DYNAREC
static inline int
page_index(page_t *p)
{
    return ((uintptr_t)p - (uintptr_t)pages) / sizeof(page_t);
}


void
page_add_to_evict_list(page_t *p)
{
    pages[purgable_page_list_head].evict_prev = page_index(p);
    p->evict_next = purgable_page_list_head;
    p->evict_prev = 0;
    purgable_page_list_head = pages[purgable_page_list_head].evict_prev;
    purgeable_page_count++;
}


void
page_remove_from_evict_list(page_t *p)
{
    if (!page_in_evict_list(p))
	fatal("page_remove_from_evict_list: not in evict list!\n");
    if (p->evict_prev)
	pages[p->evict_prev].evict_next = p->evict_next;
    else
	purgable_page_list_head = p->evict_next;
    if (p->evict_next)
	pages[p->evict_next].evict_prev = p->evict_prev;
    p->evict_prev = EVICT_NOT_IN_LIST;
	purgeable_page_count--;
}


void
mem_write_ramb_page(uint32_t addr, uint8_t val, page_t *p)
{
    if (val != p->mem[addr & 0xfff] || codegen_in_recompile) {
	uint64_t mask = (uint64_t)1 << ((addr >> PAGE_MASK_SHIFT) & PAGE_MASK_MASK);
	int byte_offset = (addr >> PAGE_BYTE_MASK_SHIFT) & PAGE_BYTE_MASK_OFFSET_MASK;
	uint64_t byte_mask = (uint64_t)1 << (addr & PAGE_BYTE_MASK_MASK);

	p->mem[addr & 0xfff] = val;
	p->dirty_mask |= mask;
	if ((p->code_present_mask & mask) && !page_in_evict_list(p))
		page_add_to_evict_list(p);
	p->byte_dirty_mask[byte_offset] |= byte_mask;
	if ((p->byte_code_present_mask[byte_offset] & byte_mask) && !page_in_evict_list(p))
		page_add_to_evict_list(p);
    }
}


void
mem_write_ramw_page(uint32_t addr, uint16_t val, page_t *p)
{
    if (val != *(uint16_t *)&p->mem[addr & 0xfff] || codegen_in_recompile) {
	uint64_t mask = (uint64_t)1 << ((addr >> PAGE_MASK_SHIFT) & PAGE_MASK_MASK);
	int byte_offset = (addr >> PAGE_BYTE_MASK_SHIFT) & PAGE_BYTE_MASK_OFFSET_MASK;
	uint64_t byte_mask = (uint64_t)1 << (addr & PAGE_BYTE_MASK_MASK);

	if ((addr & 0xf) == 0xf)
		mask |= (mask << 1);
	*(uint16_t *)&p->mem[addr & 0xfff] = val;
	p->dirty_mask |= mask;
	if ((p->code_present_mask & mask) && !page_in_evict_list(p))
		page_add_to_evict_list(p);
	if ((addr & PAGE_BYTE_MASK_MASK) == PAGE_BYTE_MASK_MASK) {
		p->byte_dirty_mask[byte_offset+1] |= 1;
		if ((p->byte_code_present_mask[byte_offset+1] & 1) && !page_in_evict_list(p))
			page_add_to_evict_list(p);
	} else
		byte_mask |= (byte_mask << 1);

	p->byte_dirty_mask[byte_offset] |= byte_mask;

	if ((p->byte_code_present_mask[byte_offset] & byte_mask) && !page_in_evict_list(p))
		page_add_to_evict_list(p);
    }
}


void
mem_write_raml_page(uint32_t addr, uint32_t val, page_t *p)
{
    if (val != *(uint32_t *)&p->mem[addr & 0xfff] || codegen_in_recompile) {
	uint64_t mask = (uint64_t)1 << ((addr >> PAGE_MASK_SHIFT) & PAGE_MASK_MASK);
	int byte_offset = (addr >> PAGE_BYTE_MASK_SHIFT) & PAGE_BYTE_MASK_OFFSET_MASK;
	uint64_t byte_mask = (uint64_t)0xf << (addr & PAGE_BYTE_MASK_MASK);

	if ((addr & 0xf) >= 0xd)
		mask |= (mask << 1);
	*(uint32_t *)&p->mem[addr & 0xfff] = val;
	p->dirty_mask |= mask;
	p->byte_dirty_mask[byte_offset] |= byte_mask;
	if (!page_in_evict_list(p) && ((p->code_present_mask & mask) || (p->byte_code_present_mask[byte_offset] & byte_mask)))
		page_add_to_evict_list(p);
	if ((addr & PAGE_BYTE_MASK_MASK) > (PAGE_BYTE_MASK_MASK-3)) {
		uint32_t byte_mask_2 = 0xf >> (4 - (addr & 3));

		p->byte_dirty_mask[byte_offset+1] |= byte_mask_2;
		if ((p->byte_code_present_mask[byte_offset+1] & byte_mask_2) && !page_in_evict_list(p))
			page_add_to_evict_list(p);
	}
    }
}
#else
void
mem_write_ramb_page(uint32_t addr, uint8_t val, page_t *p)
{
#ifdef USE_DYNAREC
    if (val != p->mem[addr & 0xfff] || codegen_in_recompile) {
#else
    if (val != p->mem[addr & 0xfff]) {
#endif
	uint64_t mask = (uint64_t)1 << ((addr >> PAGE_MASK_SHIFT) & PAGE_MASK_MASK);
	p->dirty_mask[(addr >> PAGE_MASK_INDEX_SHIFT) & PAGE_MASK_INDEX_MASK] |= mask;
	p->mem[addr & 0xfff] = val;
    }
}


void
mem_write_ramw_page(uint32_t addr, uint16_t val, page_t *p)
{
#ifdef USE_DYNAREC
    if (val != *(uint16_t *)&p->mem[addr & 0xfff] || codegen_in_recompile) {
#else
    if (val != *(uint16_t *)&p->mem[addr & 0xfff]) {
#endif
	uint64_t mask = (uint64_t)1 << ((addr >> PAGE_MASK_SHIFT) & PAGE_MASK_MASK);
	if ((addr & 0xf) == 0xf)
		mask |= (mask << 1);
	p->dirty_mask[(addr >> PAGE_MASK_INDEX_SHIFT) & PAGE_MASK_INDEX_MASK] |= mask;
	*(uint16_t *)&p->mem[addr & 0xfff] = val;
    }
}


void
mem_write_raml_page(uint32_t addr, uint32_t val, page_t *p)
{
#ifdef USE_DYNAREC
    if (val != *(uint32_t *)&p->mem[addr & 0xfff] || codegen_in_recompile) {
#else
    if (val != *(uint32_t *)&p->mem[addr & 0xfff]) {
#endif
	uint64_t mask = (uint64_t)1 << ((addr >> PAGE_MASK_SHIFT) & PAGE_MASK_MASK);
	if ((addr & 0xf) >= 0xd)
		mask |= (mask << 1);
	p->dirty_mask[(addr >> PAGE_MASK_INDEX_SHIFT) & PAGE_MASK_INDEX_MASK] |= mask;
	*(uint32_t *)&p->mem[addr & 0xfff] = val;
    }
}
#endif


void
mem_write_ram(uint32_t addr, uint8_t val, void *priv)
{
#ifdef ENABLE_MEM_LOG
    if ((addr >= 0xa0000) && (addr <= 0xbffff))
	mem_log("Write B       %02X to   %08X\n", val, addr);
#endif
    addwritelookup(mem_logical_addr, addr);
    mem_write_ramb_page(addr, val, &pages[addr >> 12]);
}


void
mem_write_ramw(uint32_t addr, uint16_t val, void *priv)
{
#ifdef ENABLE_MEM_LOG
    if ((addr >= 0xa0000) && (addr <= 0xbffff))
	mem_log("Write W     %04X to   %08X\n", val, addr);
#endif
    addwritelookup(mem_logical_addr, addr);
    mem_write_ramw_page(addr, val, &pages[addr >> 12]);
}


void
mem_write_raml(uint32_t addr, uint32_t val, void *priv)
{
#ifdef ENABLE_MEM_LOG
    if ((addr >= 0xa0000) && (addr <= 0xbffff))
	mem_log("Write L %08X to   %08X\n", val, addr);
#endif
    addwritelookup(mem_logical_addr, addr);
    mem_write_raml_page(addr, val, &pages[addr >> 12]);
}


static uint8_t
mem_read_remapped(uint32_t addr, void *priv)
{
    if ((addr >= (mem_size * 1024)) && (addr < ((mem_size + 384) * 1024)))
	addr = 0xA0000 + (addr - (mem_size * 1024));
    addreadlookup(mem_logical_addr, addr);
    return ram[addr];
}


static uint16_t
mem_read_remappedw(uint32_t addr, void *priv)
{
    if ((addr >= (mem_size * 1024)) && (addr < ((mem_size + 384) * 1024)))
	addr = 0xA0000 + (addr - (mem_size * 1024));
    addreadlookup(mem_logical_addr, addr);
    return *(uint16_t *)&ram[addr];
}


static uint32_t
mem_read_remappedl(uint32_t addr, void *priv)
{
    if ((addr >= (mem_size * 1024)) && (addr < ((mem_size + 384) * 1024)))
	addr = 0xA0000 + (addr - (mem_size * 1024));
    addreadlookup(mem_logical_addr, addr);
    return *(uint32_t *)&ram[addr];
}


static void
mem_write_remapped(uint32_t addr, uint8_t val, void *priv)
{
    uint32_t oldaddr = addr;
    if ((addr >= (mem_size * 1024)) && (addr < ((mem_size + 384) * 1024)))
	addr = 0xA0000 + (addr - (mem_size * 1024));
    addwritelookup(mem_logical_addr, addr);
    mem_write_ramb_page(addr, val, &pages[oldaddr >> 12]);
}


static void
mem_write_remappedw(uint32_t addr, uint16_t val, void *priv)
{
    uint32_t oldaddr = addr;
    if ((addr >= (mem_size * 1024)) && (addr < ((mem_size + 384) * 1024)))
	addr = 0xA0000 + (addr - (mem_size * 1024));
    addwritelookup(mem_logical_addr, addr);
    mem_write_ramw_page(addr, val, &pages[oldaddr >> 12]);
}


static void
mem_write_remappedl(uint32_t addr, uint32_t val, void *priv)
{
    uint32_t oldaddr = addr;
    if ((addr >= (mem_size * 1024)) && (addr < ((mem_size + 384) * 1024)))
	addr = 0xA0000 + (addr - (mem_size * 1024));
    addwritelookup(mem_logical_addr, addr);
    mem_write_raml_page(addr, val, &pages[oldaddr >> 12]);
}


uint8_t
mem_read_bios(uint32_t addr, void *priv)
{
    uint8_t ret = 0xff;

    addr &= 0x000fffff;

    if ((addr >= biosaddr) && (addr <= (biosaddr + biosmask)))
	ret = rom[addr - biosaddr];

    return ret;
}


uint16_t
mem_read_biosw(uint32_t addr, void *priv)
{
    uint16_t ret = 0xffff;

    addr &= 0x000fffff;

    if ((addr >= biosaddr) && (addr <= (biosaddr + biosmask)))
	ret = *(uint16_t *)&rom[addr - biosaddr];

    return ret;
}


uint32_t
mem_read_biosl(uint32_t addr, void *priv)
{
    uint32_t ret = 0xffffffff;

    addr &= 0x000fffff;

    if ((addr >= biosaddr) && (addr <= (biosaddr + biosmask)))
	ret = *(uint32_t *)&rom[addr - biosaddr];

    return ret;
}


void
mem_write_null(uint32_t addr, uint8_t val, void *p)
{
}


void
mem_write_nullw(uint32_t addr, uint16_t val, void *p)
{
}


void
mem_write_nulll(uint32_t addr, uint32_t val, void *p)
{
}


void
mem_invalidate_range(uint32_t start_addr, uint32_t end_addr)
{
    uint64_t mask;
#ifdef USE_NEW_DYNAREC
    page_t *p;

    start_addr &= ~PAGE_MASK_MASK;
    end_addr = (end_addr + PAGE_MASK_MASK) & ~PAGE_MASK_MASK;        

    for (; start_addr <= end_addr; start_addr += (1 << PAGE_MASK_SHIFT)) {
	if ((start_addr >> 12) >= pages_sz)
		continue;

	mask = (uint64_t)1 << ((start_addr >> PAGE_MASK_SHIFT) & PAGE_MASK_MASK);

	p = &pages[start_addr >> 12];

	p->dirty_mask |= mask;
	if ((p->code_present_mask & mask) && !page_in_evict_list(p))
		page_add_to_evict_list(p);
    }
#else
    uint32_t cur_addr;
    start_addr &= ~PAGE_MASK_MASK;
    end_addr = (end_addr + PAGE_MASK_MASK) & ~PAGE_MASK_MASK;	

    for (; start_addr <= end_addr; start_addr += (1 << PAGE_MASK_SHIFT)) {
	mask = (uint64_t)1 << ((start_addr >> PAGE_MASK_SHIFT) & PAGE_MASK_MASK);

	/* Do nothing if the pages array is empty or DMA reads/writes to/from PCI device memory addresses
	   may crash the emulator. */
	cur_addr = (start_addr >> 12);
	if (cur_addr < pages_sz)
		pages[cur_addr].dirty_mask[(start_addr >> PAGE_MASK_INDEX_SHIFT) & PAGE_MASK_INDEX_MASK] |= mask;
    }
#endif
}


static __inline int
mem_mapping_read_allowed(uint32_t flags, int state, int exec)
{
    int smm_state = state >> MEM_STATE_SMM_SHIFT;

    if (in_smm && ((smm_state & MEM_READ_MASK) != MEM_READ_NORMAL))
	state = smm_state;

    switch (state & MEM_READ_MASK) {
	case MEM_READ_DISABLED:
		return 0;

	case MEM_READ_ANY:
		return 1;

	/* On external and 0 mappings without ROMCS. */
	case MEM_READ_EXTERNAL:
		return !(flags & MEM_MAPPING_INTERNAL) && !(flags & MEM_MAPPING_ROMCS);

	/* On external and 0 mappings with ROMCS. */
	case MEM_READ_ROMCS:
		return !(flags & MEM_MAPPING_INTERNAL) && (flags & MEM_MAPPING_ROMCS);

	/* On any external mappings. */
	case MEM_READ_EXTANY:
		return !(flags & MEM_MAPPING_INTERNAL);

	case MEM_READ_EXTERNAL_EX:
		if (exec)
			return !(flags & MEM_MAPPING_EXTERNAL);
		else
			return !(flags & MEM_MAPPING_INTERNAL);

	case MEM_READ_INTERNAL:
		return !(flags & MEM_MAPPING_EXTERNAL);

	default:
		fatal("mem_mapping_read_allowed : bad state %x\n", state);
    }

    return 0;
}


static __inline int
mem_mapping_write_allowed(uint32_t flags, int state)
{
    int smm_state = state >> MEM_STATE_SMM_SHIFT;

    if (in_smm && ((smm_state & MEM_WRITE_MASK) != MEM_WRITE_NORMAL))
	state = smm_state;

    switch (state & MEM_WRITE_MASK) {
	case MEM_WRITE_DISABLED:
		return 0;

	case MEM_WRITE_ANY:
		return 1;

	/* On external and 0 mappings without ROMCS. */
	case MEM_WRITE_EXTERNAL:
		return !(flags & MEM_MAPPING_INTERNAL) && !(flags & MEM_MAPPING_ROMCS);

	/* On external and 0 mappings with ROMCS. */
	case MEM_WRITE_ROMCS:
		return !(flags & MEM_MAPPING_INTERNAL) && (flags & MEM_MAPPING_ROMCS);

	/* On any external mappings. */
	case MEM_WRITE_EXTANY:
		return !(flags & MEM_MAPPING_INTERNAL);

	case MEM_WRITE_INTERNAL:
		return !(flags & MEM_MAPPING_EXTERNAL);

	default:
		fatal("mem_mapping_write_allowed : bad state %x\n", state);
    }

    return 0;
}


void
mem_mapping_recalc(uint64_t base, uint64_t size)
{
    mem_mapping_t *map = base_mapping.next;
    uint64_t c;

    if (! size) return;

    /* Clear out old mappings. */
    for (c = base; c < base + size; c += MEM_GRANULARITY_SIZE) {
	read_mapping[c >> MEM_GRANULARITY_BITS] = NULL;
	write_mapping[c >> MEM_GRANULARITY_BITS] = NULL;
	_mem_exec[c >> MEM_GRANULARITY_BITS] = NULL;
    }

    /* Walk mapping list. */
    while (map != NULL) {
	/*In range?*/
	if (map->enable && (uint64_t)map->base < ((uint64_t)base + (uint64_t)size) && ((uint64_t)map->base + (uint64_t)map->size) > (uint64_t)base) {
		uint64_t start = (map->base < base) ? map->base : base;
		uint64_t end   = (((uint64_t)map->base + (uint64_t)map->size) < (base + size)) ? ((uint64_t)map->base + (uint64_t)map->size) : (base + size);
		if (start < map->base)
			start = map->base;

		for (c = start; c < end; c += MEM_GRANULARITY_SIZE) {
			if ((map->read_b || map->read_w || map->read_l) &&
			     mem_mapping_read_allowed(map->flags, _mem_state[c >> MEM_GRANULARITY_BITS], 0)) {
#ifdef ENABLE_MEM_LOG
				if ((start >= 0xa0000) && (start <= 0xbffff))
					mem_log("Read allowed: %08X (mapping for %08X)\n", map, start);
#endif
				read_mapping[c >> MEM_GRANULARITY_BITS] = map;
			}
			if (map->exec &&
			     mem_mapping_read_allowed(map->flags, _mem_state[c >> MEM_GRANULARITY_BITS], 1)) {
#ifdef ENABLE_MEM_LOG
				if ((start >= 0xa0000) && (start <= 0xbffff))
					mem_log("Exec allowed: %08X (mapping for %08X)\n", map, start);
#endif
				_mem_exec[c >> MEM_GRANULARITY_BITS] = map->exec + (c - map->base);
			}
			if ((map->write_b || map->write_w || map->write_l) &&
			     mem_mapping_write_allowed(map->flags, _mem_state[c >> MEM_GRANULARITY_BITS])) {
#ifdef ENABLE_MEM_LOG
				if ((start >= 0xa0000) && (start <= 0xbffff))
					mem_log("Write allowed: %08X (mapping for %08X)\n", map, start);
#endif
				write_mapping[c >> MEM_GRANULARITY_BITS] = map;
			}
		}
	}
	map = map->next;
    }

    flushmmucache_cr3();
}


void
mem_mapping_del(mem_mapping_t *map)
{
    mem_mapping_t *ptr;

    /* Disable the entry. */
    mem_mapping_disable(map);

    /* Zap it from the list. */
    for (ptr = &base_mapping; ptr->next != NULL; ptr = ptr->next) {
	if (ptr->next == map) {
		ptr->next = map->next;
		break;
	}
    }
}


void
mem_mapping_add(mem_mapping_t *map,
		uint32_t base, 
		uint32_t size, 
		uint8_t  (*read_b)(uint32_t addr, void *p),
		uint16_t (*read_w)(uint32_t addr, void *p),
		uint32_t (*read_l)(uint32_t addr, void *p),
		void (*write_b)(uint32_t addr, uint8_t  val, void *p),
		void (*write_w)(uint32_t addr, uint16_t val, void *p),
		void (*write_l)(uint32_t addr, uint32_t val, void *p),
		uint8_t *exec,
		uint32_t fl,
		void *p)
{
    mem_mapping_t *dest = &base_mapping;

    /* Add mapping to the end of the list.*/
    while (dest->next)
	dest = dest->next;
    dest->next = map;
    map->prev = dest;

    if (size)
	map->enable  = 1;
      else
	map->enable  = 0;
    map->base    = base;
    map->size    = size;
    map->read_b  = read_b;
    map->read_w  = read_w;
    map->read_l  = read_l;
    map->write_b = write_b;
    map->write_w = write_w;
    map->write_l = write_l;
    map->exec    = exec;
    map->flags   = fl;
    map->p       = p;
    map->dev     = NULL;
    map->next    = NULL;

    mem_mapping_recalc(map->base, map->size);
}


void
mem_mapping_do_recalc(mem_mapping_t *map)
{
    mem_mapping_recalc(map->base, map->size);
}


void
mem_mapping_set_handler(mem_mapping_t *map,
			uint8_t  (*read_b)(uint32_t addr, void *p),
			uint16_t (*read_w)(uint32_t addr, void *p),
			uint32_t (*read_l)(uint32_t addr, void *p),
			void (*write_b)(uint32_t addr, uint8_t  val, void *p),
			void (*write_w)(uint32_t addr, uint16_t val, void *p),
			void (*write_l)(uint32_t addr, uint32_t val, void *p))
{
    map->read_b  = read_b;
    map->read_w  = read_w;
    map->read_l  = read_l;
    map->write_b = write_b;
    map->write_w = write_w;
    map->write_l = write_l;

    mem_mapping_recalc(map->base, map->size);
}


void
mem_mapping_set_addr(mem_mapping_t *map, uint32_t base, uint32_t size)
{
    /* Remove old mapping. */
    map->enable = 0;
    mem_mapping_recalc(map->base, map->size);

    /* Set new mapping. */
    map->enable = 1;
    map->base = base;
    map->size = size;

    mem_mapping_recalc(map->base, map->size);
}


void
mem_mapping_set_exec(mem_mapping_t *map, uint8_t *exec)
{
    map->exec = exec;

    mem_mapping_recalc(map->base, map->size);
}


void
mem_mapping_set_p(mem_mapping_t *map, void *p)
{
    map->p = p;
}


void
mem_mapping_set_dev(mem_mapping_t *map, void *p)
{
    map->dev = p;
}


void
mem_mapping_disable(mem_mapping_t *map)
{
    map->enable = 0;

    mem_mapping_recalc(map->base, map->size);
}


void
mem_mapping_enable(mem_mapping_t *map)
{
    map->enable = 1;

    mem_mapping_recalc(map->base, map->size);
}


void
mem_set_mem_state_common(int smm, uint32_t base, uint32_t size, int state)
{
    uint32_t c;

    for (c = 0; c < size; c += MEM_GRANULARITY_SIZE) {
	if (smm)
		_mem_state[(c + base) >> MEM_GRANULARITY_BITS] = (_mem_state[(c + base) >> MEM_GRANULARITY_BITS] & 0x00ff) | ((state & 0xff) << 8);
	else
		_mem_state[(c + base) >> MEM_GRANULARITY_BITS] = (_mem_state[(c + base) >> MEM_GRANULARITY_BITS] & 0xff00) | (state & 0xff);
#ifdef ENABLE_MEM_LOG
	if (((c + base) >= 0xa0000) && ((c + base) <= 0xbffff))
		mem_log("Set mem state for block at %08X to %02X\n", c + base, state);
#endif
    }

    mem_mapping_recalc(base, size);
}


void
mem_set_mem_state(uint32_t base, uint32_t size, int state)
{
    mem_set_mem_state_common(0, base, size, state);
}


void
mem_set_mem_state_smm(uint32_t base, uint32_t size, int state)
{
    mem_set_mem_state_common(1, base, size, state);
}


void
mem_add_bios(void)
{
    int temp_cpu_type, temp_cpu_16bitbus = 1;

    if (AT) {
	temp_cpu_type = machines[machine].cpu[cpu_manufacturer].cpus[cpu_effective].cpu_type;
	temp_cpu_16bitbus = (temp_cpu_type == CPU_286 || temp_cpu_type == CPU_386SX || temp_cpu_type == CPU_486SLC || temp_cpu_type == CPU_IBM386SLC || temp_cpu_type == CPU_IBM486SLC );
    }

    if (biosmask > 0x1ffff) {
	/* 256k+ BIOS'es only have low mappings at E0000-FFFFF. */
	mem_mapping_add(&bios_mapping, 0xe0000, 0x20000,
			mem_read_bios,mem_read_biosw,mem_read_biosl,
			mem_write_null,mem_write_nullw,mem_write_nulll,
			&rom[0x20000], MEM_MAPPING_EXTERNAL|MEM_MAPPING_ROM|MEM_MAPPING_ROMCS, 0);

	mem_set_mem_state(0x0e0000, 0x20000,
			  MEM_READ_ROMCS | MEM_WRITE_ROMCS);
    } else {
	mem_mapping_add(&bios_mapping, biosaddr, biosmask + 1,
			mem_read_bios,mem_read_biosw,mem_read_biosl,
			mem_write_null,mem_write_nullw,mem_write_nulll,
			rom, MEM_MAPPING_EXTERNAL|MEM_MAPPING_ROM|MEM_MAPPING_ROMCS, 0);

	mem_set_mem_state(biosaddr, biosmask + 1,
			  MEM_READ_ROMCS | MEM_WRITE_ROMCS);
    }

    if (AT) {
	mem_mapping_add(&bios_high_mapping, biosaddr | (temp_cpu_16bitbus ? 0x00f00000 : 0xfff00000), biosmask + 1,
			mem_read_bios,mem_read_biosw,mem_read_biosl,
			mem_write_null,mem_write_nullw,mem_write_nulll,
			rom, MEM_MAPPING_EXTERNAL|MEM_MAPPING_ROM|MEM_MAPPING_ROMCS, 0);

	mem_set_mem_state(biosaddr | (temp_cpu_16bitbus ? 0x00f00000 : 0xfff00000), biosmask + 1,
			  MEM_READ_ROMCS | MEM_WRITE_ROMCS);
    }
}


void
mem_a20_init(void)
{
    if (AT) {
	rammask = cpu_16bitbus ? 0xefffff : 0xffefffff;
	flushmmucache();
	mem_a20_state = mem_a20_key | mem_a20_alt;
    } else {
	rammask = 0xfffff;
	flushmmucache();
	mem_a20_key = mem_a20_alt = mem_a20_state = 0;
    }
}


/* Reset the memory state. */
void
mem_reset(void)
{
    uint32_t c, m;

    m = 1024UL * mem_size;
    if (ram != NULL) {
	free(ram);
	ram = NULL;
    }
    ram = (uint8_t *)malloc(m);		/* allocate and clear the RAM block */
    memset(ram, 0x00, m);

    /*
     * Allocate the page table based on how much RAM we have.
     * We re-allocate the table on each (hard) reset, as the
     * memory amount could have changed.
     */
    if (AT) {
	if (cpu_16bitbus) {
		/* 80186/286; maximum address space is 16MB. */
		m = 4096;
	} else {
		/* 80386+; maximum address space is 4GB. */
		if (is486) {
			/* We need this since there might be BIOS execution at the end of RAM,
			   which could break the recompiler if there's not enough page elements. */
			m = 1048576;
		} else {
			m = (mem_size + 384) >> 2;
			if ((m << 2) < (mem_size + 384))
				m++;
			if (m < 4096)
				m = 4096;
		}
	}
    } else {
	/* 8088/86; maximum address space is 1MB. */
	m = 256;
    }

    /*
     * Allocate and initialize the (new) page table.
     * We only do this if the size of the page table has changed.
     */
#if DYNAMIC_TABLES
mem_log("MEM: reset: previous pages=%08lx, pages_sz=%i\n", pages, pages_sz);
#endif
    if (pages_sz != m) {
	pages_sz = m;
	if (pages) {
		free(pages);
		pages = NULL;
	}
	pages = (page_t *)malloc(m*sizeof(page_t));
#if DYNAMIC_TABLES
mem_log("MEM: reset: new pages=%08lx, pages_sz=%i\n", pages, pages_sz);
#endif

#if DYNAMIC_TABLES
	/* Allocate the (new) lookup tables. */
	if (page_lookup != NULL) free(page_lookup);
	page_lookup = (page_t **)malloc(pages_sz*sizeof(page_t *));

	if (readlookup2 != NULL) free(readlookup2);
	readlookup2  = malloc(pages_sz*sizeof(uintptr_t));

	if (writelookup2 != NULL) free(writelookup2);
	writelookup2 = malloc(pages_sz*sizeof(uintptr_t));

#endif
    }

#if DYNAMIC_TABLES
    memset(page_lookup, 0x00, pages_sz * sizeof(page_t *));
#else
    memset(page_lookup, 0x00, (1 << 20) * sizeof(page_t *));
#endif

    memset(pages, 0x00, pages_sz*sizeof(page_t));

#ifdef USE_NEW_DYNAREC
    if (byte_dirty_mask) {
	free(byte_dirty_mask);
	byte_dirty_mask = NULL;
    }
    byte_dirty_mask = malloc((mem_size * 1024) / 8);
    memset(byte_dirty_mask, 0, (mem_size * 1024) / 8);

    if (byte_code_present_mask) {
	free(byte_code_present_mask);
	byte_code_present_mask = NULL;
    }
    byte_code_present_mask = malloc((mem_size * 1024) / 8);
    memset(byte_code_present_mask, 0, (mem_size * 1024) / 8);
#endif

    for (c = 0; c < pages_sz; c++) {
	pages[c].mem = &ram[c << 12];
	pages[c].write_b = mem_write_ramb_page;
	pages[c].write_w = mem_write_ramw_page;
	pages[c].write_l = mem_write_raml_page;
#ifdef USE_NEW_DYNAREC
	pages[c].evict_prev = EVICT_NOT_IN_LIST;
	pages[c].byte_dirty_mask = &byte_dirty_mask[c * 64];
	pages[c].byte_code_present_mask = &byte_code_present_mask[c * 64];
#endif
    }

    memset(_mem_exec,    0x00, sizeof(_mem_exec));

    memset(&base_mapping, 0x00, sizeof(base_mapping));

    memset(_mem_state, 0x00, sizeof(_mem_state));
    /* Set SMM states to (MEM_READ_NORMAL | MEM_WRITE_NORMAL). */
    for (c = 0; c < MEM_MAPPINGS_NO; c++)
	_mem_state[c] |= 0x4400;

    mem_set_mem_state(0x000000, (mem_size > 640) ? 0xa0000 : mem_size * 1024,
		      MEM_READ_INTERNAL | MEM_WRITE_INTERNAL);
    /* mem_set_mem_state(0x0c0000, 0x40000,
		      MEM_READ_EXTERNAL | MEM_WRITE_EXTERNAL); */
    mem_set_mem_state(0x0a0000, 0x60000,
		      MEM_READ_EXTERNAL | MEM_WRITE_EXTERNAL);

    mem_mapping_add(&ram_low_mapping, 0x00000,
		    (mem_size > 640) ? 0xa0000 : mem_size * 1024,
		    mem_read_ram,mem_read_ramw,mem_read_raml,
		    mem_write_ram,mem_write_ramw,mem_write_raml,
		    ram, MEM_MAPPING_INTERNAL, NULL);

    if (mem_size > 1024) {
	if (cpu_16bitbus && mem_size > 16256) {
		mem_set_mem_state(0x100000, (16256 - 1024) * 1024,
				  MEM_READ_INTERNAL | MEM_WRITE_INTERNAL);
		mem_mapping_add(&ram_high_mapping, 0x100000,
				((16256 - 1024) * 1024),
				mem_read_ram,mem_read_ramw,mem_read_raml,
				mem_write_ram,mem_write_ramw,mem_write_raml,
				ram + 0x100000, MEM_MAPPING_INTERNAL, NULL);
	} else {
		mem_set_mem_state(0x100000, (mem_size - 1024) * 1024,
				  MEM_READ_INTERNAL | MEM_WRITE_INTERNAL);
		mem_mapping_add(&ram_high_mapping, 0x100000,
				((mem_size - 1024) * 1024),
				mem_read_ram,mem_read_ramw,mem_read_raml,
				mem_write_ram,mem_write_ramw,mem_write_raml,
				ram + 0x100000, MEM_MAPPING_INTERNAL, NULL);
	}
    }

    /* if (mem_size > 768)
	mem_mapping_add(&ram_mid_mapping, 0xc0000, 0x40000,
			mem_read_ram,mem_read_ramw,mem_read_raml,
			mem_write_ram,mem_write_ramw,mem_write_raml,
			ram + 0xc0000, MEM_MAPPING_INTERNAL, NULL); */
    if (mem_size > 768)
	mem_mapping_add(&ram_mid_mapping, 0xa0000, 0x60000,
			mem_read_ram,mem_read_ramw,mem_read_raml,
			mem_write_ram,mem_write_ramw,mem_write_raml,
			ram + 0xa0000, MEM_MAPPING_INTERNAL, NULL);

    mem_mapping_add(&ram_remapped_mapping, mem_size * 1024, 256 * 1024,
		    mem_read_remapped,mem_read_remappedw,mem_read_remappedl,
		    mem_write_remapped,mem_write_remappedw,mem_write_remappedl,
		    ram + 0xa0000, MEM_MAPPING_INTERNAL, NULL);
    mem_mapping_disable(&ram_remapped_mapping);			
			
    mem_a20_init();

#ifdef USE_NEW_DYNAREC
    purgable_page_list_head = 0;
    purgeable_page_count = 0;
#endif
}


void
mem_init(void)
{
    /* Perform a one-time init. */
    ram = rom = NULL;
    pages = NULL;
#if DYNAMIC_TABLES
    page_lookup = NULL;
    readlookup2 = NULL;
    writelookup2 = NULL;

#else
    /* Allocate the lookup tables. */
    page_lookup = (page_t **)malloc((1<<20)*sizeof(page_t *));

    readlookup2  = malloc((1<<20)*sizeof(uintptr_t));

    writelookup2 = malloc((1<<20)*sizeof(uintptr_t));
#endif

#if FIXME
    memset(ff_array, 0xff, sizeof(ff_array));
#endif

    /* Reset the memory state. */
    mem_reset();
}


void
mem_remap_top(int kb)
{
    int c;
    uint32_t start = (mem_size >= 1024) ? mem_size : 1024;
    int offset, size = mem_size - 640;

    mem_log("MEM: remapping top %iKB (mem=%i)\n", kb, mem_size);
    if (mem_size <= 640) return;

    if (kb == 0) {
 	/* Called to disable the mapping. */
 	mem_mapping_disable(&ram_remapped_mapping);

 	return;
     }	
	
    if (size > kb)
	size = kb;

    for (c = ((start * 1024) >> 12); c < (((start + size) * 1024) >> 12); c++) {
	offset = c - ((start * 1024) >> 12);
	pages[c].mem = &ram[0xA0000 + (offset << 12)];
	pages[c].write_b = mem_write_ramb_page;
	pages[c].write_w = mem_write_ramw_page;
	pages[c].write_l = mem_write_raml_page;
#ifdef USE_NEW_DYNAREC
	pages[c].evict_prev = EVICT_NOT_IN_LIST;
	pages[c].byte_dirty_mask = &byte_dirty_mask[offset * 64];
	pages[c].byte_code_present_mask = &byte_code_present_mask[offset * 64];
#endif
    }

    mem_set_mem_state(start * 1024, size * 1024,
		      MEM_READ_INTERNAL | MEM_WRITE_INTERNAL);
    mem_mapping_set_addr(&ram_remapped_mapping, start * 1024, size * 1024);
    mem_mapping_set_exec(&ram_remapped_mapping, ram + (start * 1024));

    flushmmucache();
}


void
mem_reset_page_blocks(void)
{
    uint32_t c;

    if (pages == NULL) return;

    for (c = 0; c < pages_sz; c++) {
	pages[c].write_b = mem_write_ramb_page;
	pages[c].write_w = mem_write_ramw_page;
	pages[c].write_l = mem_write_raml_page;
#ifdef USE_NEW_DYNAREC
	pages[c].block = BLOCK_INVALID;
	pages[c].block_2 = BLOCK_INVALID;
#else
	pages[c].block[0] = pages[c].block[1] = pages[c].block[2] = pages[c].block[3] = NULL;
	pages[c].block_2[0] = pages[c].block_2[1] = pages[c].block_2[2] = pages[c].block_2[3] = NULL;
#endif
    }
}


void
mem_a20_recalc(void)
{
    int state;

    if (! AT) {
	rammask = 0xfffff;
	flushmmucache();
	mem_a20_key = mem_a20_alt = mem_a20_state = 0;

	return;
    }

    state = mem_a20_key | mem_a20_alt;
    if (state && !mem_a20_state) {
	rammask = (AT && cpu_16bitbus) ? 0xffffff : 0xffffffff;
	flushmmucache();
    } else if (!state && mem_a20_state) {
	rammask = (AT && cpu_16bitbus) ? 0xefffff : 0xffefffff;
	flushmmucache();
    }

    mem_a20_state = state;
}
