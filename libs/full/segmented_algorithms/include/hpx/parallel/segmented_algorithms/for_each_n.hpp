//  Copyright (c) 2007-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/algorithms.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::parallel {

    ///////////////////////////////////////////////////////////////////////////
    // segmented_for_each_n
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static util::detail::algorithm_result_t<ExPolicy, SegIter>
        segmented_for_each_n(Algo&& algo, ExPolicy const& policy, SegIter first,
            std::size_t count, F&& f, Proj&& proj, std::true_type)
        {
            using traits = hpx::traits::segmented_iterator_traits<SegIter>;
            using segment_iterator = typename traits::segment_iterator;
            using local_iterator_type = typename traits::local_iterator;
            using result = util::detail::algorithm_result<ExPolicy, SegIter>;

            segment_iterator sit = traits::segment(first);
            local_iterator_type beg = traits::local(first);
            local_iterator_type out = beg;

            while (count != 0)
            {
                local_iterator_type end = traits::end(sit);

                local_iterator_type seg_end;
                std::size_t n;
                if constexpr (hpx::traits::is_random_access_iterator_v<
                                  local_iterator_type>)
                {
                    std::size_t const seg =
                        static_cast<std::size_t>(std::distance(beg, end));
                    n = (std::min)(count, seg);
                    seg_end = beg;
                    std::advance(seg_end, n);
                }
                else
                {
                    seg_end = beg;
                    n = 0;
                    while (n < count && seg_end != end)
                    {
                        ++seg_end;
                        ++n;
                    }
                }

                if (n != 0)
                {
                    out = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, seg_end, f, proj);
                }

                count -= n;
                if (count == 0)
                    break;

                ++sit;
                beg = traits::begin(sit);
            }

            return result::get(traits::compose(sit, out));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static util::detail::algorithm_result_t<ExPolicy, SegIter>
        segmented_for_each_n(Algo&& algo, ExPolicy const& policy, SegIter first,
            std::size_t count, F&& f, Proj&& proj, std::false_type)
        {
            using traits = hpx::traits::segmented_iterator_traits<SegIter>;
            using segment_iterator = typename traits::segment_iterator;
            using local_iterator_type = typename traits::local_iterator;
            using result = util::detail::algorithm_result<ExPolicy, SegIter>;

            using forced_seq = std::integral_constant<bool,
                !hpx::traits::is_forward_iterator_v<SegIter>>;

            segment_iterator sit = traits::segment(first);
            local_iterator_type beg = traits::local(first);

            std::vector<future<local_iterator_type>> segments;
            segment_iterator last_sit = sit;

            while (count != 0)
            {
                local_iterator_type end = traits::end(sit);

                local_iterator_type seg_end;
                std::size_t n;
                if constexpr (hpx::traits::is_random_access_iterator_v<
                                  local_iterator_type>)
                {
                    std::size_t const seg =
                        static_cast<std::size_t>(std::distance(beg, end));
                    n = (std::min)(count, seg);
                    seg_end = beg;
                    std::advance(seg_end, n);
                }
                else
                {
                    seg_end = beg;
                    n = 0;
                    while (n < count && seg_end != end)
                    {
                        ++seg_end;
                        ++n;
                    }
                }

                if (n != 0)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit), algo,
                        policy, forced_seq(), beg, seg_end, f, proj));
                    last_sit = sit;
                }

                count -= n;
                if (count == 0)
                    break;

                ++sit;
                beg = traits::begin(sit);
            }

            return result::get(dataflow(
                [last_sit](std::vector<hpx::future<local_iterator_type>>&& r)
                    -> SegIter {
                    // handle any remote exceptions, will throw on error
                    std::list<std::exception_ptr> errors;
                    parallel::util::detail::handle_remote_exceptions<
                        ExPolicy>::call(r, errors);
                    return traits::compose(last_sit, r.back().get());
                },
                HPX_MOVE(segments)));
        }

        /// \endcond
    }    // namespace detail
}    // namespace hpx::parallel

// The segmented iterators we support all live in namespace hpx::segmented
namespace hpx::segmented {

    template <typename InIter, typename Size, typename F>
        requires(hpx::traits::is_iterator_v<InIter> &&
            hpx::traits::is_segmented_iterator_v<InIter>)
    InIter tag_invoke(hpx::for_each_n_t, InIter first, Size count, F&& f)
    {
        static_assert((hpx::traits::is_input_iterator_v<InIter>),
            "Requires at least input iterator.");

        using iterator_traits = hpx::traits::segmented_iterator_traits<InIter>;

        if (hpx::parallel::detail::is_negative(count) || count == 0)
        {
            return first;
        }

        return hpx::parallel::detail::segmented_for_each_n(
            hpx::parallel::detail::for_each<
                typename iterator_traits::local_iterator>(),
            hpx::execution::seq, first, static_cast<std::size_t>(count),
            HPX_FORWARD(F, f), hpx::identity_v, std::true_type());
    }

    template <typename ExPolicy, typename SegIter, typename Size, typename F>
        requires(hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_iterator_v<SegIter> &&
            hpx::traits::is_segmented_iterator_v<SegIter>)
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, SegIter>
    tag_invoke(
        hpx::for_each_n_t, ExPolicy&& policy, SegIter first, Size count, F&& f)
    {
        static_assert((hpx::traits::is_forward_iterator_v<SegIter>),
            "Requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        if (hpx::parallel::detail::is_negative(count) || count == 0)
        {
            using result =
                hpx::parallel::util::detail::algorithm_result<ExPolicy,
                    SegIter>;
            return result::get(HPX_MOVE(first));
        }

        using iterator_traits = hpx::traits::segmented_iterator_traits<SegIter>;

        return hpx::parallel::detail::segmented_for_each_n(
            hpx::parallel::detail::for_each<
                typename iterator_traits::local_iterator>(),
            HPX_FORWARD(ExPolicy, policy), first,
            static_cast<std::size_t>(count), HPX_FORWARD(F, f), hpx::identity_v,
            is_seq());
    }
}    // namespace hpx::segmented
