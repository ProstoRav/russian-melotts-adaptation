from __future__ import annotations

import os
import sys

import torch


MELO = "${MELOTTS_ROOT}/melo"

sys.path.insert(
    0,
    MELO,
)

import train


DEFAULT_BOUNDARIES = [
    32,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
    1600,
    1800,
    2000,
    2200,
    2400,
    2600,
    2800,
]

DEFAULT_BATCH_SIZES = [
    7,  # 32-300
    7,  # 300-400
    7,  # 400-500
    7,  # 500-600
    7,  # 600-700
    7,  # 700-800
    7,  # 800-900
    7,  # 900-1000
    7,  # 1000-1100
    6,  # 1100-1200
    6,  # 1200-1300
    5,  # 1300-1400
    5,  # 1400-1500
    5,  # 1500-1600
    4,  # 1600-1800
    4,  # 1800-2000
    3,  # 2000-2200
    3,  # 2200-2400
    3,  # 2400-2600
    2,  # 2600-2800
]


def parse_int_list(
    env_name,
    default,
):
    raw = os.environ.get(
        env_name
    )

    if not raw:
        return list(default)

    try:
        values = [
            int(x.strip())
            for x in raw.split(",")
            if x.strip()
        ]
    except Exception as exc:
        raise RuntimeError(
            f"invalid {env_name}: {raw!r}"
        ) from exc

    return values


class VariableBatchBucketSampler:
    """
    Single-GPU deterministic bucket sampler with a
    separate batch size for every length interval.

    It intentionally mirrors the shuffle/padding behavior
    of MeloTTS DistributedBucketSampler where relevant.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        if num_replicas is None:
            num_replicas = 1

        if rank is None:
            rank = 0

        if (
            int(num_replicas) != 1
            or int(rank) != 0
        ):
            raise RuntimeError(
                "VariableBatchBucketSampler v1 "
                "supports the current single-GPU "
                "production contract only: "
                f"num_replicas={num_replicas}, "
                f"rank={rank}"
            )

        self.dataset = dataset
        self.lengths = dataset.lengths
        self.shuffle = bool(shuffle)

        self.boundaries = parse_int_list(
            "MELOTTS_VAR_BUCKET_BOUNDARIES",
            DEFAULT_BOUNDARIES,
        )

        self.batch_sizes = parse_int_list(
            "MELOTTS_VAR_BUCKET_BATCH_SIZES",
            DEFAULT_BATCH_SIZES,
        )

        if len(self.boundaries) < 2:
            raise RuntimeError(
                "need at least two boundaries"
            )

        if any(
            b <= a
            for a, b in zip(
                self.boundaries[:-1],
                self.boundaries[1:],
            )
        ):
            raise RuntimeError(
                "bucket boundaries must be "
                "strictly increasing"
            )

        if (
            len(self.batch_sizes)
            != len(self.boundaries) - 1
        ):
            raise RuntimeError(
                "batch-size count must equal "
                "len(boundaries)-1: "
                f"{len(self.batch_sizes)} vs "
                f"{len(self.boundaries)-1}"
            )

        if any(
            x <= 0
            for x in self.batch_sizes
        ):
            raise RuntimeError(
                "all bucket batch sizes must be > 0"
            )

        self.epoch = 0

        self.buckets = [
            []
            for _ in self.batch_sizes
        ]

        outside = []

        for idx, length in enumerate(
            self.lengths
        ):
            bucket_idx = self._bucket_index(
                length
            )

            if bucket_idx is None:
                outside.append(
                    (idx, length)
                )
            else:
                self.buckets[
                    bucket_idx
                ].append(idx)

        if outside:
            preview = outside[:20]

            raise RuntimeError(
                "dataset samples outside variable "
                "bucket coverage: "
                f"count={len(outside)}, "
                f"preview={preview}"
            )

        self.num_samples_per_bucket = []
        self.num_batches_per_bucket = []

        for bucket, batch_size_i in zip(
            self.buckets,
            self.batch_sizes,
        ):
            n = len(bucket)

            if n == 0:
                padded = 0
                batches = 0
            else:
                rem = (
                    batch_size_i
                    - (n % batch_size_i)
                ) % batch_size_i

                padded = n + rem
                batches = (
                    padded
                    // batch_size_i
                )

            self.num_samples_per_bucket.append(
                padded
            )

            self.num_batches_per_bucket.append(
                batches
            )

        self.total_size = sum(
            self.num_samples_per_bucket
        )

        self.total_batches = sum(
            self.num_batches_per_bucket
        )

        print(
            "VariableBucket boundaries:",
            self.boundaries,
        )

        print(
            "VariableBucket batch_sizes:",
            self.batch_sizes,
        )

        print(
            "VariableBucket raw_counts:",
            [
                len(x)
                for x in self.buckets
            ],
        )

        print(
            "VariableBucket padded_counts:",
            self.num_samples_per_bucket,
        )

        print(
            "VariableBucket batches:",
            self.num_batches_per_bucket,
        )

        print(
            "VariableBucket totals: "
            f"dataset={len(self.lengths)}, "
            f"sampler_samples={self.total_size}, "
            f"padding={self.total_size-len(self.lengths)}, "
            f"batches={self.total_batches}"
        )


    def _bucket_index(
        self,
        length,
    ):
        for i, (
            low,
            high,
        ) in enumerate(
            zip(
                self.boundaries[:-1],
                self.boundaries[1:],
            )
        ):
            if (
                low < length
                and length <= high
            ):
                return i

        return None


    def set_epoch(
        self,
        epoch,
    ):
        self.epoch = int(
            epoch
        )


    def __len__(
        self,
    ):
        return self.total_batches


    def __iter__(
        self,
    ):
        g = torch.Generator()

        g.manual_seed(
            self.epoch
        )

        batches = []

        for (
            bucket,
            batch_size_i,
            padded_n,
        ) in zip(
            self.buckets,
            self.batch_sizes,
            self.num_samples_per_bucket,
        ):
            if not bucket:
                continue

            if self.shuffle:
                local_ids = torch.randperm(
                    len(bucket),
                    generator=g,
                ).tolist()
            else:
                local_ids = list(
                    range(
                        len(bucket)
                    )
                )

            rem = (
                padded_n
                - len(bucket)
            )

            if rem:
                local_ids = (
                    local_ids
                    + local_ids
                    * (
                        rem
                        // len(bucket)
                    )
                    + local_ids[
                        :(
                            rem
                            % len(bucket)
                        )
                    ]
                )

            if len(local_ids) != padded_n:
                raise RuntimeError(
                    "variable sampler padding "
                    "invariant failed"
                )

            if (
                len(local_ids)
                % batch_size_i
                != 0
            ):
                raise RuntimeError(
                    "variable sampler divisibility "
                    "invariant failed"
                )

            for start in range(
                0,
                len(local_ids),
                batch_size_i,
            ):
                ids = local_ids[
                    start:
                    start + batch_size_i
                ]

                batches.append(
                    [
                        bucket[j]
                        for j in ids
                    ]
                )

        if self.shuffle:
            order = torch.randperm(
                len(batches),
                generator=g,
            ).tolist()

            batches = [
                batches[i]
                for i in order
            ]

        if len(batches) != self.total_batches:
            raise RuntimeError(
                "variable sampler batch-count "
                "invariant failed"
            )

        return iter(
            batches
        )


#
# Replace only the symbol already imported by train.py.
#
train.DistributedBucketSampler = (
    VariableBatchBucketSampler
)


if __name__ == "__main__":
    print(
        "MELOTTS VARIABLE BUCKET RUNNER V1"
    )

    train.run()
