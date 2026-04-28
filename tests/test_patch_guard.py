from nightrunner.patch_guard import validate_changed_files


def test_editable_file_allowed() -> None:
    result = validate_changed_files(
        changed_files=["train.py"],
        editable_files=["train.py"],
        protected_files=[],
        allow_new_files=False,
        allow_dependency_changes=False,
        new_files=[],
    )
    assert result["ok"] is True


def test_non_editable_file_rejected() -> None:
    result = validate_changed_files(
        changed_files=["prepare.py"],
        editable_files=["train.py"],
        protected_files=[],
        allow_new_files=False,
        allow_dependency_changes=False,
        new_files=[],
    )
    assert result["ok"] is False
    assert any(v["type"] == "not_editable" for v in result["violations"])


def test_protected_file_rejected() -> None:
    result = validate_changed_files(
        changed_files=[".env"],
        editable_files=[".env", "train.py"],
        protected_files=[".env"],
        allow_new_files=False,
        allow_dependency_changes=False,
        new_files=[],
    )
    assert result["ok"] is False
    assert any(v["type"] == "protected_file" for v in result["violations"])


def test_editable_directory_allowed() -> None:
    result = validate_changed_files(
        changed_files=["src/model.py"],
        editable_files=["src/"],
        protected_files=[],
        allow_new_files=False,
        allow_dependency_changes=False,
        new_files=[],
    )
    assert result["ok"] is True


def test_protected_priority_over_editable() -> None:
    result = validate_changed_files(
        changed_files=["src/secret.py"],
        editable_files=["src/"],
        protected_files=["src/"],
        allow_new_files=False,
        allow_dependency_changes=False,
        new_files=[],
    )
    assert result["ok"] is False
    assert any(v["type"] == "protected_file" for v in result["violations"])
