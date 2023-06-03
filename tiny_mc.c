/* Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)"
 * 1 W Point Source Heating in Infinite Isotropic Scattering Medium
 * http://omlc.ogi.edu/software/mc/tiny_mc.c
 *
 * Adaptado para CP2014, Nicolas Wolovick
 */

#define _XOPEN_SOURCE 500 // M_PI
#define INT_PREC 6
#define TEST_SIZE (1 << 27)
#ifndef NAME
#define NAME 0
#endif

#include "params.h"
#include "wtime.h"

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define foo4random() (arc4random() % ((unsigned)RAND_MAX + 1))

char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";


// global state, heat and heat square in each

static float heat[48 * SHELLS];
static float heat2[48 * SHELLS];
static float maptable[32768];


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

inline float mFast_Log2(float val)
{
    union {
        float val;
        int32_t x;
    } u = { val };
    register float log_2 = (float)(((u.x >> 23) & 255) - 128);
    u.x &= ~(255 << 23);
    u.x += 127 << 23;
    log_2 += ((-0.3358287811f) * u.val + 2.0f) * u.val - 0.65871759316667f;
    return (log_2);
}


inline float very_very_fast_logf(float x)
{
    // using the taylor expansion serie of log(1+x)
    // log(1+x) = x - x^2/2 + x^3/3 - x^4/4 + x^5/5
    float xx = x - 1;
    float xx2 = xx * xx;
    float xx3 = xx * xx2;
    float xx4 = xx * xx3;
    float xx5 = xx * xx4;
    float xx6 = xx * xx5;

    return xx - xx2 / 2 + xx3 / 3 - xx4 / 4 + xx5 / 5 - xx6 / 6;
}

inline float very_very_fast_logf_avx2(float x)
{
    __m256 xx = _mm256_set1_ps(x - 1);
    __m256 xx2 = _mm256_mul_ps(xx, xx);
    __m256 xx3 = _mm256_mul_ps(xx2, xx);
    __m256 xx4 = _mm256_mul_ps(xx2, xx2);
    __m256 xx5 = _mm256_mul_ps(xx3, xx2);
    __m256 xx6 = _mm256_mul_ps(xx3, xx3);

    __m256 term1 = _mm256_sub_ps(xx, _mm256_mul_ps(xx2, _mm256_set1_ps(0.5f)));
    __m256 term2 = _mm256_add_ps(xx3, _mm256_mul_ps(xx5, _mm256_set1_ps(0.2f)));
    __m256 term3 = _mm256_sub_ps(xx4, _mm256_mul_ps(xx3, _mm256_set1_ps(0.25f)));
    __m256 term4 = _mm256_add_ps(_mm256_mul_ps(xx5, _mm256_set1_ps(0.2f)), _mm256_mul_ps(xx4, _mm256_set1_ps(0.2f)));
    __m256 term5 = _mm256_sub_ps(_mm256_mul_ps(xx6, _mm256_set1_ps(0.16666666666666666666666666666667f)), _mm256_mul_ps(xx5, _mm256_set1_ps(0.16666666666666666666666666666667f)));

    __m256 result = _mm256_add_ps(term1, term2);
    result = _mm256_add_ps(result, term3);
    result = _mm256_add_ps(result, term4);
    result = _mm256_add_ps(result, term5);

    float res[8];
    _mm256_storeu_ps(res, result);

    return res[0];
}


/*********************************
 *******SIMPSON'S 3/8 RULE********
 ********************************/

inline float f(float x)
{
    return -1 / x;
}


float simpson38(float a)
{
    int n = INT_PREC;
    float h = (1 - a) / n;
    float x;
    float sum = 0;
    for (int i = 1; i < n; i++) {
        x = a + i * h;
        if (i % 3 == 0)
            sum += 2 * f(x);
        else
            sum += 3 * f(x);
    }
    return (3 * h / 8) * (f(a) - 1 + sum);
}
/*
int main()
{
    // test simpson38 against fast_logf
    very_fast_srand(SEED);
    // speed test comparison
    // start timer
    // print test size
    printf("Test size: %d\n", TEST_SIZE);
    double start = wtime();

    for (int i = 0; i < TEST_SIZE; i++) {
        int rn = very_fast_rand();
        if (rn == 0)
            rn = 1;

        float x = rn / (float)32767;
        float z = simpson38(x);
    }
    // end timer
    double end = wtime();
    printf("Simpson38 time: %f\n", end - start);

    // start timer
    start = wtime();
    for (int i = 0; i < TEST_SIZE; i++) {
        int rn = very_fast_rand();
        if (rn == 0)
            rn = 1;

        float x = rn / (float)32767;
        float z = logf(x);
    }
    // end timer
    end = wtime();
    printf("logf time: %f\n", end - start);

    // start timer
    start = wtime();
    for (int i = 0; i < TEST_SIZE; i++) {
        int rn = very_fast_rand();
        if (rn == 0)
            rn = 1;

        float x = rn / (float)32767;
        float z = fast_logf(x);
    }
    // end timer
    end = wtime();
    printf("fast_logf time: %f\n", end - start);

    // start timer
    start = wtime();
    for (int i = 0; i < TEST_SIZE; i++) {
        int rn = very_fast_rand();
        if (rn == 0)
            rn = 1;

        float x = rn / (float)32767;
        float z = very_fast_logf(x);
    }
    // end timer
    end = wtime();
    printf("very_fast_logf time: %f\n", end - start);

    // start timer
    start = wtime();
    for (int i = 0; i < TEST_SIZE; i++) {
        int rn = very_fast_rand();
        if (rn == 0)
            rn = 1;

        float x = rn / (float)32767;
        float z = very_very_fast_logf(x);
    }
    // end timer
    end = wtime();
    printf("very_very_fast_logf time: %f\n", end - start);

    // start timer
    start = wtime();
    for (int i = 0; i < TEST_SIZE; i++) {
        int rn = very_fast_rand();
        if (rn == 0)
            rn = 1;

        float x = rn / (float)32767;
        float z = very_very_fast_logf_avx2(x);
    }
    // end timer
    end = wtime();
    printf("very_very_fast_logf_avx2 time: %f\n", end - start);

    // start timer
    start = wtime();
    for (int i = 0; i < TEST_SIZE; i++) {
        int rn = very_fast_rand();
        if (rn == 0)
            rn = 1;

        float x = rn / (float)32767;
        float z = very_very_fast_logf_avx2_2(x);
    }

    // end timer
    end = wtime();
    printf("very_very_fast_logf_avx2_2 time: %f\n", end - start);


    // check correction
    for (int i = 0; i < 10; i++) {
        int rnn = very_fast_rand();
        if (rnn == 0)
            rnn = 1;

        float xx = rnn / (float)32767;
        float zz = simpson38(xx);
        float yy = logf(xx);
        float ww = fast_logf(xx);
        float vv = very_fast_logf(xx);
        float uu = very_very_fast_logf(xx);
        float ss = very_very_fast_logf_avx2(xx);
        float tt = very_very_fast_logf_avx2_2(xx);
        printf("X= %f, logf: %f, fast_logf: %f, very_fast_logf: %f, simpson38: %f, taylor: %f, avx2: %f, avx2_2: %f\n", xx, yy, ww, vv, zz, uu, ss, tt);
    }


    //

    return 0;
}*/

inline __m256 fast_sqrt_intr(__m256 x)
{
    return _mm256_sqrt_ps(x);
}


// create maptable for values between 0 and rand() in range of [0,2^15-1]/2^15


void create_maptable()
{
    for (int i = 0; i < 32768; i++) {
        maptable[i] = very_very_fast_logf_avx2(i / (float)32767);
    }
}

float get_maptable(int i)
{
    return maptable[i];
}


/***
 * Photon
 ***/


static void photon(int tid)
{
    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);
    const __m256 albedo_v = _mm256_set1_ps(albedo);
    const __m256 shells_per_mfp_v = _mm256_set1_ps(shells_per_mfp);

    // create a 8-vector of position in x,y,z

    __m256 x_v = _mm256_setzero_ps();
    __m256 y_v = _mm256_setzero_ps();
    __m256 z_v = _mm256_setzero_ps();

    // craete a 8-vector of speed in u,v,w

    __m256 u_v = _mm256_setzero_ps();
    __m256 v_v = _mm256_setzero_ps();
    __m256 w_v = _mm256_set1_ps(1.0f);


    // both photons have the same initial weight

    __m256 weights_v = _mm256_set1_ps(1.0f);


    for (;;) {
        /* move */

        // 2 different random times for both photons
        float t_s[8] = { -get_maptable(very_fast_rand()),
                         -get_maptable(very_fast_rand()),
                         -get_maptable(very_fast_rand()),
                         -get_maptable(very_fast_rand()),
                         -get_maptable(very_fast_rand()),
                         -get_maptable(very_fast_rand()),
                         -get_maptable(very_fast_rand()),
                         -get_maptable(very_fast_rand()) };

        /*
        x += t * u;
        y += t * v;
        z += t * w;
        */
        // using avx2 to do the above calculation
        __m256 t_v = _mm256_loadu_ps(t_s);

        x_v = _mm256_fmadd_ps(t_v, u_v, x_v);
        y_v = _mm256_fmadd_ps(t_v, v_v, y_v);
        z_v = _mm256_fmadd_ps(t_v, w_v, z_v);

        // get max between x_v y_v z_v vertically and save it to max_v


        x_v = _mm256_mul_ps(x_v, x_v);
        y_v = _mm256_mul_ps(y_v, y_v);
        z_v = _mm256_mul_ps(z_v, z_v);


        // do xyz_v[0]^2 + xyz_v[1]^2 + xyz_v[2]^2 and save it in a float using avx2


        // k1 and k2 are the distance to the origin of each photons
        // float k1 = xyz_v2[0] + xyz_v2[1] + xyz_v2[2];
        // float k2 = xyz_v2[4] + xyz_v2[5] + xyz_v2[6];

        //__m256 k_v = x_v + y_v + z_v;


        /* absorb */
        __m256 sqrt_k_v = _mm256_sqrt_ps(x_v + y_v + z_v);
        __m256 shell_v = _mm256_mul_ps(sqrt_k_v, shells_per_mfp_v);

        //
        // per lane on shell_v if shell_v > SHELLS - 1 then shell_v = SHELLS - 1
        __m256 shell_indexes = _mm256_min_ps(shell_v, _mm256_set1_ps(SHELLS - 1));

        // transform shell_indexes to int and save it in shell_indexes_s

        int shell_indexes_s[8] = { (int)shell_indexes[0],
                                   (int)shell_indexes[1],
                                   (int)shell_indexes[2],
                                   (int)shell_indexes[3],
                                   (int)shell_indexes[4],
                                   (int)shell_indexes[5],
                                   (int)shell_indexes[6],
                                   (int)shell_indexes[7] };


        // calculate the new heat of each shell
        __m256 tmp = _mm256_mul_ps(weights_v, _mm256_sub_ps(_mm256_set1_ps(1.0f), albedo_v));
        __m256 tmp_2 = _mm256_mul_ps(tmp, tmp);


        __m256 heat_v1 = _mm256_set_ps(heat[tid * SHELLS + shell_indexes_s[0]],
                                       heat[tid * SHELLS + shell_indexes_s[1]],
                                       heat[tid * SHELLS + shell_indexes_s[2]],
                                       heat[tid * SHELLS + shell_indexes_s[3]],
                                       heat[tid * SHELLS + shell_indexes_s[4]],
                                       heat[tid * SHELLS + shell_indexes_s[5]],
                                       heat[tid * SHELLS + shell_indexes_s[6]],
                                       heat[tid * SHELLS + shell_indexes_s[7]]);

        __m256 heat_v2 = _mm256_set_ps(heat[tid * SHELLS + shell_indexes_s[0]],
                                       heat[tid * SHELLS + shell_indexes_s[1]],
                                       heat[tid * SHELLS + shell_indexes_s[2]],
                                       heat[tid * SHELLS + shell_indexes_s[3]],
                                       heat[tid * SHELLS + shell_indexes_s[4]],
                                       heat[tid * SHELLS + shell_indexes_s[5]],
                                       heat[tid * SHELLS + shell_indexes_s[6]],
                                       heat[tid * SHELLS + shell_indexes_s[7]]);


        heat_v1 = _mm256_add_ps(heat_v1, tmp);
        heat_v2 = _mm256_add_ps(heat_v2, tmp_2);
        weights_v = _mm256_mul_ps(weights_v, albedo_v);

        // create a dictionary of updated shells
        int updated_shells[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };


        for (int i = 0; i < 8; i++) {
            // check if shell_indexes_s[i] is in updated_shells
            bool found = false;
            for (int j = 0; j < 8; j++) {
                if (updated_shells[j] == shell_indexes_s[i]) {
                    found = true;
                    break;
                }
            }
            if (found) {
                heat[tid * SHELLS + shell_indexes_s[i]] += tmp[i];
                heat2[tid * SHELLS + shell_indexes_s[i]] += tmp_2[i];

                break;
            } else {
                heat[tid * SHELLS + shell_indexes_s[i]] = heat_v1[i];
                heat2[tid * SHELLS + shell_indexes_s[i]] = heat_v2[i];
                updated_shells[i] = shell_indexes_s[i];
            }
        }


        /*
        heat[shell_index1] += tmp1;
        heat[shell_index2] += tmp2;
        heat2[shell_index1] += tmp1 * tmp1;
        heat2[shell_index2] += tmp2 * tmp2;
        weight1 *= albedo;
        weight2 *= albedo;*/

        /* New direction, rejection method */


        __m256 xi_v1 = _mm256_set_ps(2.0f * very_fast_rand() / (float)32768 - 1.0f,
                                     2.0f * very_fast_rand() / (float)32768 - 1.0f,
                                     2.0f * very_fast_rand() / (float)32768 - 1.0f,
                                     2.0f * very_fast_rand() / (float)32768 - 1.0f,
                                     2.0f * very_fast_rand() / (float)32768 - 1.0f,
                                     2.0f * very_fast_rand() / (float)32768 - 1.0f,
                                     2.0f * very_fast_rand() / (float)32768 - 1.0f,
                                     2.0f * very_fast_rand() / (float)32768 - 1.0f);

        __m256 xi_v2 = _mm256_set_ps(2.0f * very_fast_rand() / (float)32768 - 1.0f,
                                     2.0f * very_fast_rand() / (float)32768 - 1.0f,
                                     2.0f * very_fast_rand() / (float)32768 - 1.0f,
                                     2.0f * very_fast_rand() / (float)32768 - 1.0f,
                                     2.0f * very_fast_rand() / (float)32768 - 1.0f,
                                     2.0f * very_fast_rand() / (float)32768 - 1.0f,
                                     2.0f * very_fast_rand() / (float)32768 - 1.0f,
                                     2.0f * very_fast_rand() / (float)32768 - 1.0f);


        t_v = _mm256_add_ps(_mm256_mul_ps(xi_v1, xi_v1), _mm256_mul_ps(xi_v2, xi_v2));


        __m256 mask_v = _mm256_cmp_ps(t_v, _mm256_set1_ps(1.0f), _CMP_GT_OQ);

        do {
            // per lane if mask_v is true, then calculate new xi_v1[i] and xi_v2[i]
            // else do nothing
            xi_v1 = _mm256_blendv_ps(xi_v1, _mm256_set1_ps(2.0f * very_fast_rand() / (float)32768 - 1.0f), mask_v);
            xi_v2 = _mm256_blendv_ps(xi_v2, _mm256_set1_ps(2.0f * very_fast_rand() / (float)32768 - 1.0f), mask_v);
            // recalculate t_v
            t_v = _mm256_add_ps(_mm256_mul_ps(xi_v1, xi_v1), _mm256_mul_ps(xi_v2, xi_v2));
            // recalculate mask_v
            mask_v = _mm256_cmp_ps(t_v, _mm256_set1_ps(1.0f), _CMP_GT_OQ);


        } while (_mm256_movemask_ps(mask_v) != 0);

        u_v = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), t_v), _mm256_set1_ps(1.0f));
        v_v = _mm256_mul_ps(xi_v1, _mm256_sqrt_ps(_mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(u_v, u_v)), t_v)));
        w_v = _mm256_mul_ps(xi_v2, _mm256_sqrt_ps(_mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(u_v, u_v)), t_v)));

        // get horizontal max of weights_v
        //
        float arr[8];
        _mm256_storeu_ps(arr, weights_v);
        // max of arr
        float maxv = arr[0];
        for (int i = 1; i < 8; i++) {
            if (arr[i] > maxv) {
                maxv = arr[i];
            }
        }


        if (maxv < 0.001f) { /* roulette */
            if (very_fast_rand() / (float)32768 > 0.1f)
                break;
            weights_v = _mm256_div_ps(weights_v, _mm256_set1_ps(0.1f));
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
    // start maptable
    create_maptable();

    float totalheat[SHELLS] = { 0.0f };
    float totalheat2[SHELLS] = { 0.0f };
    /*
        #pragma omp parallel private(heat,heat2) shared(totalheat1,totalheat2)
        {
            #pragma omp for schedule(dynamic, 1000) nowait
            for (unsigned int i = 0; i < PHOTONS / 8; ++i) {
                photon();
            }
            #pragma omp critical
            {
                totalheat1 += heat;
                totalheat2 += heat2;
            }

        }*/

    // declare heat and heat2


    // get number of threads
    int nthreads = omp_get_max_threads();
    printf("nthreads = %d\n", nthreads);

    // declare heat and heat2 for each thread


#pragma omp parallel for shared(heat, heat2)
    for (unsigned int i = 0; i < PHOTONS / 8; ++i) {
        // get current thread
        int tid = omp_get_thread_num();
        photon(tid);
    }

    // reduce heat and heat2


    // time the following for
    double start2 = wtime();
    for (int i = 0; i < nthreads; i++) {
        for (int j = 0; j < SHELLS; j++) {
            totalheat[j] += heat[i * SHELLS + j];
            totalheat2[j] += heat2[i * SHELLS + j];
        }
    }
    double end2 = wtime();
    printf("time for reduction = %lf\n", end2 - start2);


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
               totalheat[i] / t / (i * i + i + 1.0 / 3.0),
               fast_sqrt(totalheat2[i] - totalheat[i] * totalheat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    }
    printf("# extra\t%12.5f\n", totalheat[SHELLS - 1] / PHOTONS);


    // print to csv file to compare later
    FILE* fp;
    // set name of file based on NAME
    char filename[100];
    sprintf(filename, "heat%i.csv", NAME);
    fp = fopen(filename, "w");
    for (unsigned int i = 0; i < SHELLS - 1; ++i) {
        fprintf(fp, "%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
                heat[i] / t / (i * i + i + 1.0 / 3.0),
                fast_sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    }

    fclose(fp);


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
