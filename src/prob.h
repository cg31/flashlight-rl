#pragma once

#include <af/array.h>
#include <af/random.h>

inline int multinomial(const af::array &probs)
{
    auto u = af::randu(probs.dims(), probs.type());
    auto e = -af::log(u);
    auto s = probs / e;

    af::array val, idx;
    af::max(val, idx, s, 0);

    return idx.scalar<unsigned>();
}
