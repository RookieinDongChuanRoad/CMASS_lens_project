# Pipeline Comparison Report

## Scope

- 当前实现与参考实现均在 `key_tests` 隔离工作区内运行。
- 两套实现统一读取 `/Users/liurongfu/Work/CMASS_lens_project/data` 下的数据。
- `smoke` 仅用于链路打通检查，不作为科学比较结果。
- `compare` 是正式长跑对比模式，本轮固定为 `10000` 步并统一丢弃前 `2000` 步样本。

## Parameter Order Notes

- Current order: `mu5_0, beta5, xi5, sigma5, mu_gamma_0, beta_gamma, xi_gamma, sigma_gamma, mu_zs, sigma_zs, theta0, loga`
- Reference order: `mu5_0, beta5, xi5, sigma5, mu_gamma_0, beta_gamma, xi_gamma, sigma_gamma, mu_zs, sigma_zs, loga, theta0`
- 参考实现最后两维是 `loga, theta0`；corner notebook 会在读取后重排为当前实现顺序。

## Run Summary

| Implementation | Profile | Mode | Requested Steps | Raw Steps | Discard | Flat Samples | Median Log Prob | Chain Path |
|----------------|---------|------|-----------------|-----------|---------|--------------|-----------------|------------|
| current | sersic | smoke | 4 | 4 | 0 | 96 | -114.937248 | `/Users/liurongfu/Work/CMASS_lens_project/key_tests/output/current/sersic/smoke/sersic/20260308_223004_sersic_smoke/chain.h5` |
| reference | sersic | smoke | 4 | 4 | 0 | 96 | -105.222657 | `/Users/liurongfu/Work/CMASS_lens_project/key_tests/output/reference/sersic/smoke/reference_chain.h5` |
| current | devauc | smoke | 4 | 4 | 0 | 96 | -103.687987 | `/Users/liurongfu/Work/CMASS_lens_project/key_tests/output/current/devauc/smoke/devauc/20260308_223020_devauc_smoke/chain.h5` |
| reference | devauc | smoke | 4 | 4 | 0 | 96 | -104.599294 | `/Users/liurongfu/Work/CMASS_lens_project/key_tests/output/reference/devauc/smoke/reference_chain.h5` |
| current | sersic | compare | 10000 | 10000 | 2000 | 192000 | -29.858172 | `/Users/liurongfu/Work/CMASS_lens_project/key_tests/output/current/sersic/compare/sersic/20260308_223036_sersic_compare/chain.h5` |
| reference | sersic | compare | 10000 | 10000 | 2000 | 192000 | -16.698972 | `/Users/liurongfu/Work/CMASS_lens_project/key_tests/output/reference/sersic/compare/reference_chain.h5` |
| current | devauc | compare | 10000 | 10000 | 2000 | 192000 | -17.772638 | `/Users/liurongfu/Work/CMASS_lens_project/key_tests/output/current/devauc/compare/devauc/20260309_014301_devauc_compare/chain.h5` |
| reference | devauc | compare | 10000 | 10000 | 2000 | 192000 | -17.233736 | `/Users/liurongfu/Work/CMASS_lens_project/key_tests/output/reference/devauc/compare/reference_chain.h5` |

## Observed Differences

- 代码实现差异：当前实现使用模块化 compiled context 与 richer metadata；参考实现仍保持旧脚本结构。
- 输入数据差异：本轮已强制统一为 `CMASS_lens_project/data`，因此正式 compare 结果主要反映实现差异，而不是数据副本差异。
- 参数顺序差异：参考实现的 `loga/theta0` 顺序与当前实现相反，已在 notebook 和报告中显式标注。

## Output Locations

- Workspace root: `/Users/liurongfu/Work/CMASS_lens_project/key_tests`
- Figures directory: `/Users/liurongfu/Work/CMASS_lens_project/key_tests/output/figures`
- Notebook: `/Users/liurongfu/Work/CMASS_lens_project/key_tests/notebooks/compare_corner.ipynb`
