#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>
#include "libpopcnt.h"


int _get_cpuid() {
#if defined(__cplusplus)
    /* C++11 thread-safe singleton */
    static const int cpuid = get_cpuid();
#else
    static int cpuid_ = -1;
    int cpuid = cpuid_;
    if (cpuid == -1)
    {
        cpuid = get_cpuid();
        __sync_val_compare_and_swap(&cpuid_, -1, cpuid);
    }
#endif
    return cpuid;
}


/*
 * Count the number of 1 bits in the data array
 * @data: An array
 * @size: Size of data in bytes
 * @cpuid: Result of the cpuid call
 */
static inline uint64_t popcnt_no_cpuid(const void* data, uint64_t size, int cpuid) {
  const uint8_t* ptr = (const uint8_t*) data;
  uint64_t cnt = 0;
  uint64_t i;

#if defined(HAVE_AVX2)

  /* AVX2 requires arrays >= 512 bytes */
  if ((cpuid & bit_AVX2) &&
      size >= 512)
  {
    align_avx2(&ptr, &size, &cnt);
    cnt += popcnt_avx2((const __m256i*) ptr, size / 32);
    ptr += size - size % 32;
    size = size % 32;
  }

#endif

#if defined(HAVE_POPCNT)

  if (cpuid & bit_POPCNT)
  {
    cnt += popcnt64_unrolled((const uint64_t*) ptr, size / 8);
    ptr += size - size % 8;
    size = size % 8;
    for (i = 0; i < size; i++)
      cnt += popcnt64(ptr[i]);

    return cnt;
  }

#endif

  /* pure integer popcount algorithm */
  for (i = 0; i < size; i++)
    cnt += popcount64(ptr[i]);

  return cnt;
}


void predict_float_256(float* user_vector,
                       float* item_vectors,
                       float user_bias,
                       float* item_biases,
                       float* out,
                       intptr_t num_items,
                       intptr_t latent_dim) {

    float* item_vector;

    __m256 x, y, prediction;
    float scalar_prediction;
    float unpacked[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    int j;

    for (int i = 0; i < num_items; i++) {

        prediction = _mm256_setzero_ps();
        scalar_prediction = item_biases[i] + user_bias;

        item_vector = item_vectors + (i * latent_dim);

        for (j = 0; j + 8 <= latent_dim; j += 8) {
            x = _mm256_load_ps(item_vector + j);
            y = _mm256_load_ps(user_vector + j);

            prediction = _mm256_fmadd_ps(x, y, prediction);
        }

        _mm256_store_ps(unpacked, prediction);

        for (int k = 0; k < 8; k++) {
            scalar_prediction += unpacked[k];
        }

        // Remainder
        for (; j < latent_dim; j++) {
            scalar_prediction += item_vector[j] * user_vector[j];
        }

        out[i] = scalar_prediction;
    }
}


int predict_xnor_256_lowdim(int32_t* user_vector,
                             int32_t* item_vectors,
                             float user_bias,
                             float* item_biases,
                             float user_norm,
                             float* item_norms,
                             float* out,
                             intptr_t num_items,
                             intptr_t latent_dim,
                             int cpuid) {
    
    int i, last_idx, item_idx;
    int32_t* item_vector;
    int total_elements = num_items * latent_dim;

    __m256i x, y, xnor;
    float scalar_prediction;
    unsigned int on_bits;
    int32_t bits[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    float max_on_bits = latent_dim * 32;

    __m256i allbits = _mm256_cmpeq_epi32(
        _mm256_setzero_si256(),
        _mm256_setzero_si256());
    
    // Repeat the user vector to fit into a 256-bit AVX2 register
    int32_t user_vector_repeated[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int k; k < 8; k++) {
        user_vector_repeated[k] = user_vector[k % latent_dim];
    }

    y = _mm256_loadu_si256(user_vector_repeated);

    for (i=0; i + 8 < total_elements; i+=8) {

        item_idx = i / latent_dim;

        item_vector = item_vectors + i;

        x = _mm256_loadu_si256(item_vector);

        // XNOR
        xnor = _mm256_xor_si256(_mm256_xor_si256(x, y), allbits);
        _mm256_store_si256(bits, xnor);

        // Bitcount
        for (int k=0; k < 8 / latent_dim; k++) {

            last_idx = item_idx + k;
            
            on_bits = popcnt_no_cpuid((const void*) (bits + k * latent_dim),
                                      latent_dim * sizeof(float),
                                      cpuid);
            scalar_prediction = (on_bits - (max_on_bits - on_bits))
                * item_norms[last_idx];
                //* user_norm 
            out[last_idx] = scalar_prediction + item_biases[last_idx];
        }
    }

    return last_idx;
}


void predict_xnor_256(int32_t* user_vector,
                      int32_t* item_vectors,
                      float user_bias,
                      float* item_biases,
                      float user_norm,
                      float* item_norms,
                      float* out,
                      intptr_t num_items,
                      intptr_t latent_dim) {

    int i = 0;
    int cpuid = _get_cpuid();

    if (latent_dim < 8) {
        i = predict_xnor_256_lowdim(user_vector,
                                    item_vectors,
                                    user_bias,
                                    item_biases,
                                    user_norm,
                                    item_norms,
                                    out,
                                    num_items,
                                    latent_dim,
                                    cpuid);
    }

    int32_t* item_vector;
    int j;

    __m256i x, y, xnor;
    float scalar_prediction;
    unsigned int on_bits;
    int32_t bits[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    float max_on_bits = latent_dim * 32;

    __m256i allbits = _mm256_cmpeq_epi32(
        _mm256_setzero_si256(),
        _mm256_setzero_si256());

    for (; i < num_items; i++) {

        item_vector = item_vectors + (i * latent_dim);
        scalar_prediction = 0;
        on_bits = 0;

        j = 0;

        for (; j + 8 <= latent_dim; j += 8) {

            // Load
            x = _mm256_load_si256(item_vector + j);
            y = _mm256_load_si256(user_vector + j);

            // XNOR
            xnor = _mm256_xor_si256(_mm256_xor_si256(x, y), allbits);
            _mm256_store_si256(bits, xnor);

            // Bitcount
            on_bits += popcnt_no_cpuid((const void*) bits,
                                       8 * sizeof(float), cpuid);
        }

        for (; j < latent_dim; j++) {
            on_bits += __builtin_popcount(~(user_vector[j] ^ item_vector[j]));
        }

        // Scaling
        scalar_prediction = (on_bits - (max_on_bits - on_bits))
            * user_norm * item_norms[i];

        // Biases
        out[i] = scalar_prediction + user_bias + item_biases[i];
    }
}
