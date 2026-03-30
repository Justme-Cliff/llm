#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct TokenizerByte {
  int vocab_size; // 256 bytes + 4 specials
  uint16_t bos_id;
  uint16_t eos_id;
  uint16_t pad_id;
  uint16_t unk_id;
} TokenizerByte;

void tokenizer_byte_init(TokenizerByte* tok);

// Encodes UTF-8 bytes. Each byte becomes one token id in [0..255].
// Output is: [BOS] + bytes(s) + [EOS]
size_t tokenizer_byte_encode_text(const TokenizerByte* tok, const char* s, uint16_t* out_tokens, size_t max_tokens);

// Decodes tokens < 256 back into bytes, ignoring special tokens.
// Converts '\\' + 'n' into newline characters for readability.
size_t tokenizer_byte_decode_text(const TokenizerByte* tok, const uint16_t* tokens, size_t n, char* out, size_t max_out);

