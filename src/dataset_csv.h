#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tokenizer_byte.h"

typedef struct DatasetTokens {
  uint16_t* tokens;  // Contiguous token ids for all rows (BOS/text/EOS per row)
  size_t token_count;
} DatasetTokens;

// Loads CSV and returns a single concatenated token stream.
// Assumes first line is header and the second line onward each has a quoted string in the first column.
// Each row becomes: BOS + bytes(text) + EOS.
int dataset_load_csv_first_col_as_text(
  const char* csv_path,
  const TokenizerByte* tok,
  DatasetTokens* out_ds
);

void dataset_free(DatasetTokens* ds);

