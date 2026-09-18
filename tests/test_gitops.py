import pytest

from modelmaker import gitops


def test_status_on_a_non_repo_directory(tmp_path):
    status = gitops.status(tmp_path)
    assert status == {"is_repo": False, "branch": None, "remote": None, "changes": [], "ahead": 0, "behind": 0}


def test_init_then_status_reports_a_repo(tmp_path):
    gitops.init(tmp_path)
    status = gitops.status(tmp_path)
    assert status["is_repo"] is True
    assert status["remote"] is None
    assert status["ahead"] == 0 and status["behind"] == 0


def test_init_is_idempotent(tmp_path):
    gitops.init(tmp_path)
    gitops.init(tmp_path)  # must not raise
    assert gitops.is_repo(tmp_path)


def test_status_lists_untracked_and_modified_files(tmp_path):
    gitops.init(tmp_path)
    (tmp_path / "a.txt").write_text("one")
    status = gitops.status(tmp_path)
    assert {"status": "??", "path": "a.txt"} in status["changes"]


def test_set_remote_then_get_it_back(tmp_path):
    gitops.init(tmp_path)
    gitops.set_remote(tmp_path, "https://github.com/example/repo.git")
    assert gitops.status(tmp_path)["remote"] == "https://github.com/example/repo.git"

    # calling again re-points the existing remote instead of failing on a duplicate
    gitops.set_remote(tmp_path, "https://github.com/example/other.git")
    assert gitops.status(tmp_path)["remote"] == "https://github.com/example/other.git"


def test_commit_clears_the_working_tree_changes(tmp_path):
    gitops.init(tmp_path)
    (tmp_path / "a.txt").write_text("one")
    gitops.commit(tmp_path, "initial commit")
    assert gitops.status(tmp_path)["changes"] == []


def test_commit_with_nothing_staged_raises(tmp_path):
    gitops.init(tmp_path)
    (tmp_path / "a.txt").write_text("one")
    gitops.commit(tmp_path, "initial commit")
    with pytest.raises(gitops.GitError):
        gitops.commit(tmp_path, "empty commit")


def test_push_without_a_remote_raises(tmp_path):
    gitops.init(tmp_path)
    (tmp_path / "a.txt").write_text("one")
    gitops.commit(tmp_path, "initial commit")
    with pytest.raises(gitops.GitError):
        gitops.push(tmp_path)
