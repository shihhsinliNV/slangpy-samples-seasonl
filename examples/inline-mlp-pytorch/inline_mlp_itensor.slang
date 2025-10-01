// SPDX-License-Identifier: Apache-2.0

import "slangpy";

[Differentiable]
float relu(float x) { return max(0.0, x); }

// Note: read-only ITensor<> prevents writing; manual sampling is done inline in render_image

// One hidden layer MLP: F -> H -> 3
// W1: (F x H), b1: (H), W2: (H x 3), b2: (3)
[Differentiable]
float3 mlp_eval(
    ITensor<float,1> x,              // [F]
    ITensor<float,2> W1, ITensor<float,1> b1,
    ITensor<float,2> W2, ITensor<float,1> b2)
{
    int F = int(W1.shape[0]);
    int H = int(W1.shape[1]);

    // Hidden
    float hidden[64]; // assume H <= 64 for this toy
    [MaxIters(64)] for (int j = 0; j < H; ++j)
    {
        float acc = b1[j];
        [MaxIters(64)] for (int i = 0; i < F; ++i)
            acc += x[i] * W1[i, j];
        hidden[j] = relu(acc);
    }

    // Output 3 (avoid dynamic indexing on float3)
    float acc0 = b2[0];
    [MaxIters(64)] for (int j = 0; j < H; ++j) acc0 += hidden[j] * W2[j, 0];
    float acc1 = b2[1];
    [MaxIters(64)] for (int j = 0; j < H; ++j) acc1 += hidden[j] * W2[j, 1];
    float acc2 = b2[2];
    [MaxIters(64)] for (int j = 0; j < H; ++j) acc2 += hidden[j] * W2[j, 2];
    return float3(acc0, acc1, acc2);
}

// Render entry: per-pixel, uses call id as pixel coordinate
[Differentiable]
float3 render_image(
    ITensor<float, 3> featureGrid,   // (Gh+1, Gw+1, F)
    ITensor<float, 2> W1,
    ITensor<float, 1> b1,
    ITensor<float, 2> W2,
    ITensor<float, 1> b2,
    no_diff float2 frameDim,
    no_diff uint2 grid_cell)
{
    float2 pixel = float2(grid_cell) + 0.5;
    float2 uv = pixel / frameDim;

    // Allocate a feature vector view of length F (reusing featureGrid shape at runtime)
    // Use a small fixed-size stack array and write via a temporary tensor view is not directly expressible here,
    // so we accumulate directly within mlp using a temporary ITensor proxy. Simpler: build a small Tensor on the fly.
    // For clarity, create a temporary 1D tensor by sampling then reading back through ITensor interface.

    // Create a scratch buffer via GradOutTensor pattern is heavy; instead we inline accumulate into a local array.
    int F = int(featureGrid.shape[2]);
    float feat[64]; // assume F <= 64 for toy

    // Manual bilinear sample into local array
    int Gh = int(featureGrid.shape[0]) - 1;
    int Gw = int(featureGrid.shape[1]) - 1;
    float gx = clamp(uv.x, 0.0, 1.0) * Gh;
    float gy = clamp(uv.y, 0.0, 1.0) * Gw;
    int x0 = int(floor(gx));
    int y0 = int(floor(gy));
    int x1 = min(x0 + 1, Gh);
    int y1 = min(y0 + 1, Gw);
    float tx = frac(gx);
    float ty = frac(gy);
    [MaxIters(64)] for (int k = 0; k < F; ++k)
    {
        float f00 = featureGrid[x0, y0, k];
        float f10 = featureGrid[x1, y0, k];
        float f01 = featureGrid[x0, y1, k];
        float f11 = featureGrid[x1, y1, k];
        float f0 = lerp(f00, f10, tx);
        float f1 = lerp(f01, f11, tx);
        feat[k] = lerp(f0, f1, ty);
    }

    // Wrap local array as a simple accessor via ITensor semantics is not available, so perform MLP inline
    int H = int(W1.shape[1]);
    float hidden[64];
    [MaxIters(64)] for (int j = 0; j < H; ++j)
    {
        float acc = b1[j];
        [MaxIters(64)] for (int i = 0; i < F; ++i)
            acc += feat[i] * W1[i, j];
        hidden[j] = relu(acc);
    }
    // Output 3 (avoid dynamic indexing on float3)
    float acc0 = b2[0];
    [MaxIters(64)] for (int j = 0; j < H; ++j) acc0 += hidden[j] * W2[j, 0];
    float acc1 = b2[1];
    [MaxIters(64)] for (int j = 0; j < H; ++j) acc1 += hidden[j] * W2[j, 1];
    float acc2 = b2[2];
    [MaxIters(64)] for (int j = 0; j < H; ++j) acc2 += hidden[j] * W2[j, 2];
    return float3(acc0, acc1, acc2);
}
