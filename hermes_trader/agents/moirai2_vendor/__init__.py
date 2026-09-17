"""Vendored Moirai-2 — dependency-reduced port of Salesforce uni2ts.

Provenance
----------
Vendored 2026-09-16 from ``SalesforceAIResearch/uni2ts`` at commit
``8062ef5a5660d2fea395fd1288ec9c397396c168`` (the exact commit the
MoiraiAgent project pins, per its gift_eval requirements). Files under
``common/``, ``module/``, ``model/moirai2/module.py`` are **verbatim
copies** with only the ``uni2ts.`` import prefix rewritten to this
package (verify with the ``scripts/`` vendor-check if available).

``model/moirai2/forecast.py`` is a **hand port** of the upstream
``Moirai2Forecast`` class (see that file's docstring for the exact
delta). The port exists because the upstream class subclasses
``lightning.LightningModule`` and ``predict()`` runs through the
gluonts predictor stack — and ``uni2ts`` itself pins
``torch>=2.1,<2.5`` / ``numpy~=1.26.0`` / ``gluonts~=0.14.3`` /
``jax[cpu]``, none of which coexist with the hermes-trader container
(torch 2.13 / numpy 2.5.2 / Python 3.13). The model *body*
(``Moirai2Module``) needs only torch / einops / jaxtyping /
huggingface_hub, so it is used directly and the predictor plumbing is
re-implemented with numpy + torch only.

Fidelity gate (hard, pre-merge): vendored ``predict`` output must match
the official ``uni2ts`` implementation to <1e-4 relative median
divergence on 20+ real logged series (reference env:
``scratch/moirai2-ref``, uni2ts @ 8062ef5, torch 2.4.1+cpu,
gluonts 0.14.x). See ``scratch/moirai2_validate.py``.

License
-------
All vendored code: Apache-2.0 (Copyright (c) 2024 Salesforce, Inc. —
each upstream file retains its SPDX header). The *pretrained weights*
``Salesforce/moirai-2.0-R-small`` are **CC-BY-NC-4.0** (non-commercial)
— same posture as the TimesFM-3 integration: the module is shadow-only,
never gates/sizes, and ships behind ``expert_select.enabled: false``.
Using the weights against the live book is a production use the
operator consciously accepts.
"""

from __future__ import annotations

from .model.moirai2.forecast import Moirai2Forecast
from .model.moirai2.module import Moirai2Module

__all__ = ["Moirai2Forecast", "Moirai2Module"]
