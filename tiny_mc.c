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


// global state, heat and heat square in each

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


double fastPow(double a, double b)
{
    union {
        double d;
        int x[2];
    } u = { a };
    u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;
    return u.d;
}

float very_fast_sqrt(float x)
{
    __m256 x_avx = _mm256_set1_ps(x);
    __m256 inv_sqrt_avx = _mm256_rsqrt_ps(x_avx);
    __m256 result_avx = _mm256_mul_ps(x_avx, inv_sqrt_avx);
    return _mm256_cvtss_f32(result_avx);
}

inline float fast_logf(float x)
{
    float y = x - 0.5f;
    float y2 = y * y;
    float y3 = y * y2;
    float y4 = y * y3;
    float y5 = y * y4;
    float y6 = y * y5;
    float y7 = y * y6;
    float y8 = y * y7;
    return y / 0.5f - y2 / (2 * 0.5f * 0.5f) + y3 / (3 * 0.5f * 0.5f * 0.5f)
        - y4 / (4 * 0.5f * 0.5f * 0.5f * 0.5f) + y5 / (5 * 0.5f * 0.5f * 0.5f * 0.5f * 0.5f)
        - y6 / (6 * 0.5f * 0.5f * 0.5f * 0.5f * 0.5f * 0.5f)
        + y7 / (7 * 0.5f * 0.5f * 0.5f * 0.5f * 0.5f * 0.5f * 0.5f)
        - y8 / (8 * 0.5f * 0.5f * 0.5f * 0.5f * 0.5f * 0.5f * 0.5f * 0.5f);
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


    /* AVX2 constants */
    const __m256 albedo_v = _mm256_set1_ps(albedo);
    const __m256 one_v = _mm256_set1_ps(1.0f);

    for (;;) {
        /* move */
        const float rand1 = very_fast_rand() / (float)32768;
        float t = -fast_logf(rand1);
        x += t * u;
        y += t * v;
        z += t * w;

        /* absorb */
        const unsigned int shell = fast_sqrt(x * x + y * y + z * z) * shells_per_mfp;

        int shell_index = shell > SHELLS - 1 ? SHELLS - 1 : shell;
        /* easy way */
        /*
        heat[shell_index] += weight * (1.0f - albedo);
        heat2[shell_index] += weight * weight * (1.0f - albedo) * (1.0f - albedo);
        weight *= albedo;
        */
        /*  */


        /*using avx2 operations*/
        __m256 heat_v = _mm256_load_ps(&heat[shell]);
        __m256 heat2_v = _mm256_load_ps(&heat2[shell]);

        __m256 weight_v = _mm256_set1_ps(weight);
        __m256 tmp_v = _mm256_sub_ps(one_v, albedo_v);
        tmp_v = _mm256_mul_ps(tmp_v, weight_v);
        __m256 tmp2_v = _mm256_mul_ps(tmp_v, tmp_v);
        // tmp_v to float

        heat_v = _mm256_add_ps(heat_v, tmp_v);
        heat2_v = _mm256_add_ps(heat2_v, tmp2_v);
        // save heat and heat2

        heat[shell_index] = _mm256_cvtss_f32(heat_v);


        heat2[shell_index] = _mm256_cvtss_f32(heat2_v);

        weight_v = _mm256_mul_ps(weight_v, albedo_v);
        weight = _mm256_cvtss_f32(weight_v);
        /*  */


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
               fast_sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    }
    printf("# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);

    return 0;
}
/*
int main(void)
{
    int N = 100;
    // test of Q_sqrt function
    printf("test of Q_sqrt function\n");
    float x = 100.0f;
    float y = 0.0f;
    y = fast_sqrt(x);
    printf("Q_sqrt(%f) = %f\n", x, y);
    return 0;
}*/
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
