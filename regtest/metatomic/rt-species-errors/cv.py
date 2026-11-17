# Minimal models for the SPECIES / SELECTED_ATOMS input checks. The values only
# have to depend on the positions: the SPECIES entries are validated before the
# model is used at all, and SELECTED_ATOMS is validated just after the model is
# run once on an empty system to discover how many properties it returns. So
# these models have to load, and to survive being called with zero atoms, but
# their values are never compared against anything.
from typing import Dict, List, Optional

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)


class SumOfSquares(torch.nn.Module):
    def __init__(self, per_atom):
        super().__init__()
        self._per_atom = per_atom

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        if "feature" not in outputs:
            return {}

        device = systems[0].positions.device if len(systems) > 0 else torch.device("cpu")
        properties = Labels("cv", torch.tensor([[0]], device=device))

        if self._per_atom:
            samples_list: List[List[int]] = []
            for s, system in enumerate(systems):
                for i in range(len(system)):
                    samples_list.append([s, i])
            samples = Labels(
                ["system", "atom"],
                torch.tensor(samples_list, device=device, dtype=torch.int32).reshape(-1, 2),
            )
            values = torch.cat(
                [(system.positions**2).sum(dim=1, keepdim=True) for system in systems]
            ).to(torch.float32)
        else:
            samples = Labels(
                "system", torch.arange(len(systems), device=device).reshape(-1, 1)
            )
            values = torch.zeros((len(systems), 1), dtype=torch.float32, device=device)
            for i, system in enumerate(systems):
                values[i, 0] = (system.positions**2).sum()

        cv = TensorMap(
            keys=Labels("_", torch.tensor([[0]], device=device)),
            blocks=[
                TensorBlock(
                    values=values,
                    samples=samples,
                    components=[],
                    properties=properties,
                )
            ],
        )

        if selected_atoms is not None and self._per_atom:
            cv = mts.slice(cv, axis="samples", selection=selected_atoms)

        return {"feature": cv}


for per_atom, path in [(False, "model.pt"), (True, "model-per-atom.pt")]:
    capabilities = ModelCapabilities(
        outputs={
            "feature": ModelOutput(sample_kind="atom" if per_atom else "system")
        },
        interaction_range=0.0,
        supported_devices=["cpu"],
        length_unit="A",
        atomic_types=[6, 8],
        dtype="float32",
    )
    cv = SumOfSquares(per_atom=per_atom)
    cv.eval()
    AtomisticModel(cv, ModelMetadata(), capabilities).save(path)
