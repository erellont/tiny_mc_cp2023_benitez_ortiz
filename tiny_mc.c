/* Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)"
 * 1 W Point Source Heating in Infinite Isotropic Scattering Medium
 * http://omlc.ogi.edu/software/mc/tiny_mc.c
 *
 * Adaptado para CP2014, Nicolas Wolovick
 */

#define _XOPEN_SOURCE 500 // M_PI

#include "params.h"
#include "wtime.h"

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define foo4random() (arc4random() % ((unsigned)RAND_MAX + 1))

char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";


// global state, heat and heat square in each shell
static float heat[SHELLS];
static float heat2[SHELLS];

static unsigned int g_seed;

// Used to seed the generator.
static inline void fast_srand(int seed)
{
    g_seed = seed;
}
// Compute a pseudorandom integer.
// Output value in range [0, 32767]
static inline int fast_rand(void)
{
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) & 0x7FFF;
}

static inline void very_fast_srand(int seed)
{
    g_seed = seed;
}

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
static inline int very_fast_rand(void)
{
    g_seed = (1103515245 * g_seed) % 4294967296;
    return (g_seed >> 16) & 0x7FFF;
}


float fast_sqrt(float x)
{
    union {
        int i;
        float x;
    } u;

    u.x = x;
    u.i = (1 << 29) + (u.i >> 1) - (1 << 22);
    return u.x;
}


/***
 * Photon
 ***/

static void photon(void)
{
    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);


    /* launch */
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    float w = 1.0f;
    float weight = 1.0f;

    for (;;) {
        float t = -logf(very_fast_rand() / (float)32768); /* move */
        x += t * u;
        y += t * v;
        z += t * w;

        unsigned int shell = fast_sqrt(x * x + y * y + z * z) * shells_per_mfp; /* absorb */
        if (shell > SHELLS - 1) {
            shell = SHELLS - 1;
        }
        heat[shell] += (1.0f - albedo) * weight;
        heat2[shell] += (1.0f - albedo) * (1.0f - albedo) * weight * weight; /* add up squares */
        weight *= albedo;

        /* New direction, rejection method */
        float xi1, xi2;
        do {
            xi1 = 2.0f * very_fast_rand() / (float)32768 - 1.0f;
            xi2 = 2.0f * very_fast_rand() / (float)32768 - 1.0f;
            t = xi1 * xi1 + xi2 * xi2;
        } while (1.0f < t);
        u = 2.0f * t - 1.0f;
        v = xi1 * fast_sqrt((1.0f - u * u) / t);
        w = xi2 * fast_sqrt((1.0f - u * u) / t);

        if (weight < 0.001f) { /* roulette */
            if (very_fast_rand() / (float)32768 > 0.1f)
                break;
            weight /= 0.1f;
        }
    }
}


/***
 * Main matter
 ***/


int main(void)
{
    // heading
    printf("# %s\n# %s\n# %s\n", t1, t2, t3);
    printf("# Scattering = %8.3f/cm\n", MU_S);
    printf("# Absorption = %8.3f/cm\n", MU_A);
    printf("# Photons    = %8d\n#\n", PHOTONS);

    // configure RNG
    very_fast_srand(SEED);
    // start timer
    double start = wtime();
    // simulation
    for (unsigned int i = 0; i < PHOTONS; ++i) {
        photon();
    }
    // stop timer
    double end = wtime();
    assert(start <= end);
    double elapsed = end - start;

    printf("# %lf seconds\n", elapsed);
    printf("# %lf K photons per second\n", 1e-3 * PHOTONS / elapsed);

    printf("# Radius\tHeat\n");
    printf("# [microns]\t[W/cm^3]\tError\n");
    float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
    for (unsigned int i = 0; i < SHELLS - 1; ++i) {
        printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
               heat[i] / t / (i * i + i + 1.0 / 3.0),
               sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    }
    printf("# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);

    return 0;
}
/*

#define N (1 << 27)

int main(void)
{
    // test

    int rnd = 0;
    double start = wtime();
    double end = wtime();

    start = wtime();
    very_fast_srand(SEED);
    for (int i = 0; i < N; ++i) {
        rnd = very_fast_rand();
    }
    end = wtime();
    printf("time for very_fast_rand %lf with N = %d, rng per second: %lf\n", end - start, N, N / (end - start));

    start = wtime();
    fast_srand(SEED);
    for (int i = 0; i < N; ++i) {
        rnd = fast_rand();
    }
    end = wtime();
    printf("time for fast_rand %lf with N = %d, rng per second: %lf\n", end - start, N, N / (end - start));


    start = wtime();
    srand(SEED);
    for (int i = 0; i < N; ++i) {
        rnd = rand();
    }
    end = wtime();
    printf("time for rand %lf with N = %d, rng per second: %lf\n", end - start, N, N / (end - start));

    return 0;
}
*/
