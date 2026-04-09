# Shared utility — canonical location is utils/cheatsheet.py
from utils.cheatsheet import *  # noqa: F401
from utils.cheatsheet import (  # noqa: F401
    Cheatsheet, DECISION_TREE_MAX_CHARS, CASE_STUDY_MAX_CHARS,
    TOTAL_RENDER_MAX_CHARS, DECISION_TREE_HEADER, CASE_STUDIES_HEADER,
    CASE_STUDY_DIVIDER, PRIOR_KNOWLEDGE_HEADER, ROADMAP_HEADER,
    _truncate,
)
from utils.case_study import _extract_title_from_text as _extract_title  # noqa: F401
