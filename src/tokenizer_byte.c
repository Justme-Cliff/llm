#include "tokenizer_byte.h"

#include <string.h>

void tokenizer_byte_init(TokenizerByte* tok) {
  if (!tok) return;
  tok->vocab_size = 260;
  tok->bos_id = 256;
  tok->eos_id = 257;
  tok->pad_id = 258;
  tok->unk_id = 259;
}

size_t tokenizer_byte_encode_text(const TokenizerByte* tok, const char* s, uint16_t* out_tokens, size_t max_tokens) {
  if (!tok || !s || !out_tokens || max_tokens == 0) return 0;
  size_t len = strlen(s);
  size_t need = 2 + len; // BOS + bytes + EOS
  if (need > max_tokens) {
    // Truncate bytes while preserving BOS/EOS.
    if (max_tokens < 2) return 0;
    len = max_tokens - 2;
  }

  size_t i = 0;
  out_tokens[i++] = tok->bos_id;
  for (size_t j = 0; j < len; j++) {
    out_tokens[i++] = (uint8_t)s[j];
  }
  out_tokens[i++] = tok->eos_id;
  return i;
}

size_t tokenizer_byte_decode_text(const TokenizerByte* tok, const uint16_t* tokens, size_t n, char* out, size_t max_out) {
  if (!tok || !tokens || !out || max_out == 0) return 0;
  size_t o = 0;
  for (size_t i = 0; i < n; i++) {
    uint16_t t = tokens[i];
    if (t == tok->bos_id || t == tok->eos_id || t == tok->pad_id || t == tok->unk_id) continue;
    if (t < 256) {
      unsigned char byte = (unsigned char)t;
      // Convert literal "\\n" to real newline.
      if (t == (uint8_t)'\\' && (i + 1) < n && tokens[i + 1] == (uint8_t)'n') {
        if (o + 1 >= max_out) break;
        out[o++] = '\n';
        i++; // skip 'n'
        continue;
      }
      if (byte < 32 || byte == 127) {
        // Keep output readable and JSON-safe.
        if (o + 1 >= max_out) break;
        out[o++] = '?';
        continue;
      }
      if (byte > 126) {
        // Avoid invalid/garbled UTF-8 in browser/JSON.
        if (o + 1 >= max_out) break;
        out[o++] = '?';
        continue;
      }
      if (o + 1 >= max_out) break;
      out[o++] = (char)byte;
    }
  }
  out[o] = '\0';
  return o;
}

