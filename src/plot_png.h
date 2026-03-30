#pragma once

// Reads a CSV loss log (step,loss,elapsed_seconds) and writes a PNG line chart.
// Returns 0 on success.
int plot_png_loss_from_csv(const char* loss_csv, const char* out_png, int width, int height);

