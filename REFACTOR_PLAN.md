# Refactoring Plan: Split `paperpipe.py` into Modules

## Current State

`paperpipe.py` is 7,220 lines containing everything: config, utilities, paper operations, CLI commands, search, PaperQA2 integration, LEANN integration, and installation logic.

## Proposed Module Structure

```
paperpipe/
├── __init__.py          # Re-exports public API for backwards compat
├── __main__.py          # Entry point: `python -m paperpipe`
├── cli.py               # Click group + all @cli.command decorators (~800 lines)
├── config.py            # Config loading, settings getters (~500 lines)
├── core.py              # Index I/O, arXiv parsing, name utilities (~400 lines)
├── paper.py             # Add/remove/update, metadata fetch, LLM generation (~700 lines)
├── search.py            # FTS5 index, grep search, audit (~600 lines)
├── paperqa.py           # PaperQA2 indexing, ask, staging (~800 lines)
├── leann.py             # LEANN index build + MCP server (~200 lines)
├── install.py           # Skills, prompts, MCP server installation (~600 lines)
└── output.py            # echo/debug helpers, quiet mode (~60 lines)
```

## Dependency Graph (must be acyclic)

```
output.py          ← no deps (only stdlib + click)
    ↑
config.py          ← output (for debug())
    ↑
core.py            ← config, output
    ↑
paper.py           ← core, config, output
    ↑
search.py          ← core, config, output, paper (for index access)
    ↑
paperqa.py         ← core, config, output, paper
    ↑
leann.py           ← core, config, output
    ↑
install.py         ← config, output
    ↑
cli.py             ← all of the above (thin dispatcher)
```

## Detailed Split Points

### 1. `output.py` (lines 44-98)
- `_debug_logger`, `_setup_debug_logging()`
- `_quiet_mode`, `set_quiet()`, `echo()`, `echo_success()`, `echo_error()`, `echo_warning()`, `echo_progress()`, `debug()`

### 2. `config.py` (lines 100-700)
- `PAPER_DB`, `PAPERS_DIR`, `INDEX_FILE` (derived from env)
- `DEFAULT_*` constants
- `_CONFIG_CACHE`, `load_config()`, `_config_get()`, `_setting_str()`, `_setting_float()`
- All `default_*()` functions (llm, embedding, pqa, leann settings)
- `_is_ollama_model_id()`, `_normalize_ollama_base_url()`, `_prepare_ollama_env()`, `_ollama_reachability_error()`
- `tag_aliases()`, `normalize_tag()`, `normalize_tags()`
- `CATEGORY_TAGS`

### 3. `core.py` (lines 700-1060)
- `_format_title_short()`, `_slugify_title()`, `_parse_authors()`, `_looks_like_pdf()`
- `_generate_local_pdf_name()`, `ensure_notes_file()`
- arXiv regex patterns, `arxiv_base_id()`, `normalize_arxiv_id()`, `_arxiv_base_from_any()`, `_index_arxiv_base_to_names()`
- `_is_safe_paper_name()`, `_resolve_paper_name_from_ref()`
- `_normalize_for_search()`, `_read_text_limited()`, `_best_line_ratio()`, `_fuzzy_text_score()`
- `ensure_db()`, `load_index()`, `save_index()`, `categories_to_tags()`
- `_VALID_REGENERATE_FIELDS`, `_parse_overwrite_option()`, `_is_arxiv_id_name()`

### 4. `paper.py` (lines 1379-2240)
- `fetch_arxiv_metadata()`, `download_pdf()`, `download_source()`
- `extract_equations_simple()`, `_extract_section_headings()`, `_extract_equation_blocks()`
- `_extract_name_from_title()`, `_generate_name_with_llm()`
- `_litellm_available()`, `_run_llm()`
- `_add_single_paper()`, `_add_local_pdf()`, `_update_existing_paper()`
- `_regenerate_one_paper()`

### 5. `search.py` (lines 3070-3960)
- `_search_grep()`, `_relativize_grep_output()`, `_parse_grep_matches()`, `_collect_grep_matches()`
- `_search_db_path()`, `_sqlite_connect()`, `_sqlite_fts5_available()`
- `_ensure_search_index_schema()`, `_set_search_index_meta()`, `_get_search_index_include_tex()`
- `_search_index_delete()`, `_search_index_upsert()`, `_search_index_rebuild()`, `_search_fts()`, `_fts5_quote_literal()`
- `_maybe_update_search_index()`, `_maybe_delete_from_search_index()`
- Audit functions: `_extract_referenced_title_from_equations()`, `_extract_suspicious_tokens_from_summary()`, `_extract_summary_title()`, `_check_boilerplate()`, `_audit_paper_dir()`, `_parse_selection_spec()`

### 6. `paperqa.py` (lines 1064-1340, 4291-5672)
- `_pillow_available()`, `_refresh_pqa_pdf_staging_dir()`
- `_extract_flag_value()`, `_paperqa_effective_paper_directory()`, `_paperqa_find_crashing_file()`
- `_paperqa_index_files_path()`, `_paperqa_load_index_files_map()`, `_paperqa_save_index_files_map()`
- `_paperqa_clear_failed_documents()`, `_paperqa_mark_failed_documents()`
- `_ModelProbeResult`, `_first_line()`, `_probe_hint()`
- `_pqa_is_noisy_stream_line()`, `_pqa_is_noisy_index_line()`, `_pqa_has_flag()`
- `_pqa_print_filtered_output_on_failure()`, `_pqa_print_filtered_index_output_on_failure()`
- `_paperqa_ask_evidence_blocks()`
- Logic from `index_cmd()` and `ask()` commands (factored into functions)

### 7. `leann.py` (lines 4143-4270, 6309-6325)
- `_leann_index_meta_path()`, `_leann_build_index()`
- `_ask_leann()`
- MCP server runner

### 8. `install.py` (lines 6326-7050)
- `_install_skill()`, `_install_prompts()`, `_install_mcp()`
- `_parse_components()`
- `_uninstall_skill()`, `_uninstall_prompts()`, `_uninstall_mcp()`

### 9. `cli.py` (entry point)
- `@click.group()` definition
- All `@cli.command()` decorators
- Each command function becomes a thin wrapper calling into the appropriate module
- Imports from all other modules

## Migration Strategy

### Phase 1: Create package structure (non-breaking)
1. Create `paperpipe/` directory
2. Move `paperpipe.py` → `paperpipe/_monolith.py` (temporary)
3. Create `paperpipe/__init__.py` that re-exports everything from `_monolith`
4. Create `paperpipe/__main__.py` with `from paperpipe._monolith import cli; cli()`
5. Update `pyproject.toml` entry point
6. **Test**: Ensure `papi` CLI still works

### Phase 2: Extract leaf modules first
1. `output.py` - no dependencies
2. `config.py` - depends only on `output`
3. Update `_monolith.py` to import from these

### Phase 3: Extract core utilities
1. `core.py` - depends on `config`, `output`
2. Update `_monolith.py`

### Phase 4: Extract feature modules
1. `paper.py`
2. `search.py`
3. `paperqa.py`
4. `leann.py`
5. `install.py`

### Phase 5: Extract CLI
1. `cli.py` - thin command wrappers
2. Delete `_monolith.py`
3. Update `__init__.py` to export public API from new modules

### Phase 6: Cleanup
1. Update imports in `test_paperpipe.py`
2. Remove any remaining re-exports that aren't needed
3. Update `pyproject.toml` if needed

## Backwards Compatibility

The `paperpipe` package will export all currently-public functions from `__init__.py`:
- `load_index`, `save_index`, `ensure_db`
- `fetch_arxiv_metadata`, `download_pdf`, `download_source`
- `normalize_arxiv_id`, `arxiv_base_id`
- `categories_to_tags`, `normalize_tag`, `normalize_tags`
- CLI entry point

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Circular imports | Strict dependency order; `output` → `config` → `core` → others |
| Breaking external imports | Re-export from `__init__.py` |
| Global state (`_CONFIG_CACHE`, `_quiet_mode`) | Keep in their respective modules; import the module, not the variable |
| Test breakage | Update test imports; run tests after each phase |
| Typos during move | Use editor refactoring tools; run tests frequently |

## Estimated Effort

- Phase 1: 30 min
- Phase 2-3: 1 hour
- Phase 4: 2 hours
- Phase 5: 1 hour
- Phase 6: 30 min

Total: ~5 hours if careful, testing after each phase.

## Decision Points

1. **Keep `paperpipe.py` at root or move to `src/paperpipe/`?**
   - Recommendation: `paperpipe/` at repo root (simpler, matches current layout)

2. **Rename internal `_` functions when extracting?**
   - Recommendation: Keep names; they're already well-named

3. **Split tests too?**
   - Recommendation: Not initially; split later if `test_paperpipe.py` becomes unwieldy
