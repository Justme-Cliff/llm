#include "dataset_csv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* util_strndup(const char* s, size_t n) {
  char* out = (char*)malloc(n + 1);
  if (!out) return NULL;
  memcpy(out, s, n);
  out[n] = '\0';
  return out;
}

static int parse_first_quoted_field(const char* line, char** out_text, size_t* out_len) {
  // Very small CSV parser: finds first " ... " on the line.
  const char* a = strchr(line, '"');
  if (!a) return -1;
  const char* b = strchr(a + 1, '"');
  if (!b) return -1;
  if (b <= a + 1) {
    *out_text = util_strndup("", 0);
    *out_len = 0;
    return 0;
  }
  size_t n = (size_t)(b - (a + 1));
  char* s = util_strndup(a + 1, n);
  if (!s) return -1;
  *out_text = s;
  *out_len = n;
  return 0;
}

int dataset_load_csv_first_col_as_text(
  const char* csv_path,
  const TokenizerByte* tok,
  DatasetTokens* out_ds
) {
  if (!csv_path || !tok || !out_ds) return -1;
  memset(out_ds, 0, sizeof(*out_ds));

  FILE* f = fopen(csv_path, "rb");
  if (!f) return -1;

  // Read full file so we can do a simple line split.
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return -1;
  }
  long sz = ftell(f);
  if (sz < 0) {
    fclose(f);
    return -1;
  }
  rewind(f);

  char* buf = (char*)malloc((size_t)sz + 1);
  if (!buf) {
    fclose(f);
    return -1;
  }
  size_t rd = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  buf[rd] = '\0';

  // Skip header line.
  char* p = buf;
  char* line_start = p;
  int line_idx = 0;

  size_t cap_tokens = 0;
  uint16_t* tokens = NULL;
  size_t token_count = 0;

  while (*p) {
    if (*p == '\n') {
      // Extract line [line_start, p).
      size_t line_len = (size_t)(p - line_start);
      if (line_idx >= 1) {
        // Parse first quoted field from all rows after the header.
        char* line = util_strndup(line_start, line_len);
        if (!line) break;

        char* text = NULL;
        size_t text_len = 0;
        if (parse_first_quoted_field(line, &text, &text_len) == 0) {
          // Encode text (BOS + bytes + EOS).
          size_t need = 2 + text_len;
          if (token_count + need > cap_tokens) {
            size_t new_cap = cap_tokens == 0 ? (need * 4) : cap_tokens;
            while (token_count + need > new_cap) new_cap *= 2;
            uint16_t* nt = (uint16_t*)realloc(tokens, new_cap * sizeof(uint16_t));
            if (!nt) {
              free(text);
              free(line);
              break;
            }
            tokens = nt;
            cap_tokens = new_cap;
          }
          size_t written = 0;
          tokens[token_count + written++] = tok->bos_id;
          for (size_t i = 0; i < text_len; i++) {
            tokens[token_count + written++] = (uint8_t)text[i];
          }
          tokens[token_count + written++] = tok->eos_id;
          token_count += written;
        }
        free(text);
        free(line);
      }
      // line_idx counts from 0 for header; line_idx>=1 are data rows.
      line_idx++;
      p++;
      line_start = p;
      continue;
    }
    p++;
  }

  // Handle last line without '\n'
  if (p != line_start) {
    if (line_idx >= 1) {
      size_t line_len = (size_t)(p - line_start);
      if (line_len > 0) {
        char* line = util_strndup(line_start, line_len);
        if (line) {
          char* text = NULL;
          size_t text_len = 0;
          if (parse_first_quoted_field(line, &text, &text_len) == 0) {
            size_t need = 2 + text_len;
            if (token_count + need > cap_tokens) {
              size_t new_cap = cap_tokens == 0 ? (need * 4) : cap_tokens;
              while (token_count + need > new_cap) new_cap *= 2;
              uint16_t* nt = (uint16_t*)realloc(tokens, new_cap * sizeof(uint16_t));
              if (nt) {
                tokens = nt;
                cap_tokens = new_cap;
              }
            }
            if (token_count + need <= cap_tokens) {
              size_t written = 0;
              tokens[token_count + written++] = tok->bos_id;
              for (size_t i = 0; i < text_len; i++) {
                tokens[token_count + written++] = (uint8_t)text[i];
              }
              tokens[token_count + written++] = tok->eos_id;
              token_count += written;
            }
          }
          free(text);
          free(line);
        }
      }
    }
  }

  free(buf);

  if (token_count == 0) {
    free(tokens);
    return -1;
  }

  out_ds->tokens = tokens;
  out_ds->token_count = token_count;
  return 0;
}

void dataset_free(DatasetTokens* ds) {
  if (!ds) return;
  free(ds->tokens);
  ds->tokens = NULL;
  ds->token_count = 0;
}

