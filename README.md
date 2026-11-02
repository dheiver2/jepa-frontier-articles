# Artigos de fronteira sobre JEPA (preprints)

Status: **publicados no Zenodo como preprints** (acesso aberto, CC BY 4.0). Os resultados são de experimentos piloto em pequena escala; o protocolo completo de cada artigo está descrito no texto e pendente. Só o artigo 1 tem um braço experimental completo (simulação, 45 execuções).

## DOIs no Zenodo

| Artigo | DOI |
|---|---|
| `artigo_01_collapse` | [10.5281/zenodo.23890644](https://doi.org/10.5281/zenodo.23890644) |
| `artigo_02_world_model` | [10.5281/zenodo.23890647](https://doi.org/10.5281/zenodo.23890647) |
| `artigo_03_causal` | [10.5281/zenodo.23890650](https://doi.org/10.5281/zenodo.23890650) |
| `artigo_04_language_reasoning` | [10.5281/zenodo.23890654](https://doi.org/10.5281/zenodo.23890654) |
| `artigo_05_hierarchical` | [10.5281/zenodo.23890658](https://doi.org/10.5281/zenodo.23890658) |
| `artigo_06_scaling` | [10.5281/zenodo.23890660](https://doi.org/10.5281/zenodo.23890660) |
| `artigo_07_uncertainty` | [10.5281/zenodo.23890662](https://doi.org/10.5281/zenodo.23890662) |
| `artigo_08_ecg` | [10.5281/zenodo.23890664](https://doi.org/10.5281/zenodo.23890664) |
| `artigo_09_shift` | [10.5281/zenodo.23890666](https://doi.org/10.5281/zenodo.23890666) |
| `artigo_10_probing` | [10.5281/zenodo.23890668](https://doi.org/10.5281/zenodo.23890668) |

Autor: Dheiver Francisco Santos, Independent Researcher, ORCID [0000-0002-8599-9436](https://orcid.org/0000-0002-8599-9436).

## Pendências para versões futuras

1. **Executar os experimentos** do protocolo completo de cada artigo, com no mínimo 5 sementes e intervalo de confiança, e publicar como nova versão no Zenodo.
2. **Confirmar as DOIs** das referências do autor em `refs.bib`, que vieram do registro ORCID.
3. **Checagem de novidade:** rodar busca de literatura, porque a área de JEPA se move rápido.
4. **Veículo e limite** de páginas, conforme as instruções autorais de cada revista ou conferência.

## Arquivos

| Arquivo | Tema |
|---|---|
| `artigo_01_collapse.tex` | Colapso e regularização (SIGReg, VICReg, EMA) |
| `artigo_02_world_model.tex` | JEPA como modelo de mundo com MPC |
| `artigo_03_causal.tex` | Consistência intervencional em simulações causais |
| `artigo_04_language_reasoning.tex` | Language JEPA para raciocínio |
| `artigo_05_hierarchical.tex` | JEPA hierárquico para documentos longos |
| `artigo_06_scaling.tex` | Leis de escala: JEPA, MAE e contrastivo |
| `artigo_07_uncertainty.tex` | Futuros multimodais e incerteza calibrada |
| `artigo_08_ecg.tex` | Pré-treino JEPA em ECG com alinhamento a laudos |
| `artigo_09_shift.tex` | Mudança de distribuição e adaptação contínua (PT-BR e EN) |
| `artigo_10_probing.tex` | Sondas nos espaços latentes contra verdade-terreno simulada |

## Como compilar

Com Tectonic instalado (`brew install tectonic`):

```bash
tectonic artigo_01_collapse.tex
```

Os rascunhos usam `\bibliography{refs}` e `\bibliographystyle{IEEEtran}`. Cada compilação gera o PDF com a referência resolvida, mas as seções `[PENDING]` continuam visíveis no PDF por design.
