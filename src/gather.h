#pragma once

#include <vector>
#include <assert.h>
#include <algorithm>
#include <numeric>

#include "flashlight/fl/common/Init.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/optim/optim.h"

inline fl::Variable gather(const fl::Variable &in, const af::array &idx)
{
    af::array result(idx.dims());
    auto idims = in.dims();

    for (dim_t i = 0; i < idims[1]; i++)
    {
        af::array currInCol = in.array()(af::span, i);
        af::array currIdxCol = idx(af::span, i);
        result(af::span, i) = currInCol(currIdxCol);
    }

    auto gradFunc = [idx, idims](std::vector<fl::Variable> &inputs, const fl::Variable &gradOutput)
    {
        auto grad = af::constant(0, idims, inputs[0].type());

        for (dim_t i = 0; i < idims[1]; i++)
        {
            af::array currInCol = gradOutput.array()(af::span, i);
            af::array currIdxCol = idx(af::span, i );
            grad(currIdxCol, i) = currInCol;
        }

        inputs[0].addGrad(fl::Variable(grad, false));
    };

    return fl::Variable(result, {in.withoutData()}, gradFunc);
}

inline fl::Variable max(const fl::Variable &in, int dim)
{
    af::array result, idx;
    af::max(result, idx, in.array(), dim);
    auto idims = in.dims();

    auto gradFunc = [idx, idims](std::vector<fl::Variable> &inputs, const fl::Variable &gradOutput)
    {
        auto grad = af::constant(0, idims, inputs[0].type());

        for (dim_t i = 0; i < idims[1]; i++)
        {
            af::array currInCol = gradOutput.array()(af::span, i);
            af::array currIdxCol = idx(af::span, i);
            grad(currIdxCol, i) = currInCol;
        }

        inputs[0].addGrad(fl::Variable(grad, false));
    };

    return fl::Variable(result, {in.withoutData()}, gradFunc);
}
