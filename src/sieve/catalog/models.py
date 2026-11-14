from __future__ import annotations

from typing import Annotated, Literal, Self

import astropy.units as u
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from sieve.contracts.models import Finite, Identifier, Text, check_finite
from sieve.provenance.models import Digest


class CatalogContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal["sieve.catalog.v1"] = "sieve.catalog.v1"

    @model_validator(mode="after")
    def finite(self) -> Self:
        check_finite(self.model_dump(mode="python"))
        return self


class FeatureSpec(CatalogContract):
    name: Identifier
    column: Text
    unit: Text
    description: Text
    lower: Finite
    upper: Finite
    error_column: Text | None = None
    uncertainty_kind: Literal["reported_standard_deviation"] | None = None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        u.Unit(self.unit, parse_strict="raise")
        if self.lower >= self.upper:
            raise ValueError("feature support lower must be less than upper")
        if bool(self.error_column) != bool(self.uncertainty_kind):
            raise ValueError("error column requires explicit reported uncertainty semantics")
        return self


class SplitGroup(CatalogContract):
    namespace: Identifier
    column: Text


class QualitySpec(CatalogContract):
    column: Text
    accepted_values: tuple[Text, ...] = Field(min_length=1)
    definition_ref: Text


class CatalogManifest(CatalogContract):
    """Frozen, explicit column mapping; units and support are declarations, not proof."""

    path: Text
    format: Literal["csv", "jsonl", "parquet"]
    sha256: Digest
    provider: Text
    dataset: Text
    release: Text
    data_kind: Literal["REAL_ARCHIVE", "SYNTHETIC", "USER_SUPPLIED_UNVERIFIED"]
    source_uri: Text
    request_parameters: dict[str, JsonValue]
    citations: tuple[Text, ...] = Field(min_length=1)
    identity_namespace: Identifier
    id_column: Text
    ra_column: Text
    dec_column: Text
    coordinate_frame: Literal["icrs"]
    coordinate_unit: Literal["deg"]
    features: tuple[FeatureSpec, ...] = Field(min_length=1, max_length=64)
    split_groups: tuple[SplitGroup, ...] = Field(min_length=1)
    split_policy_ref: Text
    support_domain: Text
    selection_description: Text
    quality: QualitySpec | None = None
    calibration_refs: tuple[Text, ...] = ()
    limitations: tuple[Text, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            raise ValueError("duplicate feature name")
        if len({f.column for f in self.features}) != len(self.features):
            raise ValueError("duplicate source feature column; do not repeat an input as evidence")
        if len({g.namespace for g in self.split_groups}) != len(self.split_groups):
            raise ValueError("duplicate split-group namespace")
        forbidden = {
            "truth",
            "ground_truth",
            "truth_record",
            "target_label",
            "class_label",
            "benchmark_label",
        }
        structural = {self.id_column, self.ra_column, self.dec_column}
        structural.update(g.column for g in self.split_groups)
        if self.quality:
            structural.add(self.quality.column)
        for feature in self.features:
            if feature.name.lower() in forbidden or feature.column.lower() in forbidden:
                raise ValueError("evaluator truth/label columns are forbidden ranking features")
            if feature.column in structural:
                raise ValueError(
                    "identity, split, coordinate and quality columns cannot be features"
                )
        return self

    def columns(self) -> tuple[str, ...]:
        columns = [self.id_column, self.ra_column, self.dec_column]
        columns += [f.column for f in self.features]
        columns += [f.error_column for f in self.features if f.error_column]
        columns += [g.column for g in self.split_groups]
        if self.quality:
            columns.append(self.quality.column)
        return tuple(dict.fromkeys(columns))


class CatalogConfig(CatalogContract):
    review_budget: Annotated[int, Field(ge=0, le=1000)] = 20
    per_region: Annotated[int, Field(ge=1)] = 5
    region_width_deg: Annotated[int, Field(ge=1, le=90)] = 10
    training_rows: Annotated[int, Field(ge=8, le=100_000)] = 4096
    trees: Annotated[int, Field(ge=1, le=500)] = 100
    batch_rows: Annotated[int, Field(ge=1, le=100_000)] = 1024
    max_rows_per_input: Annotated[int, Field(ge=1)] = 1_000_000
    max_input_bytes: Annotated[int, Field(ge=1)] = 2_000_000_000
    max_work_bytes: Annotated[int, Field(ge=1)] = 4_000_000_000
    max_line_bytes: Annotated[int, Field(ge=1024)] = 1_000_000
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 42
    export_json_array: bool = False
