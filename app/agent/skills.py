import importlib.util
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SkillRunner = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]


class SkillLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedSkill:
    name: str
    description: str
    instructions: str
    metadata: dict[str, Any]
    run: SkillRunner


class SkillLoader:
    def __init__(self, skills_dir: str | Path) -> None:
        self.skills_dir = Path(skills_dir)

    def load_all(self) -> dict[str, LoadedSkill]:
        if not self.skills_dir.exists():
            return {}
        skills: dict[str, LoadedSkill] = {}
        for path in sorted(
            item
            for item in self.skills_dir.iterdir()
            if item.is_dir() and not item.name.startswith((".", "__"))
        ):
            skill = self.load(path)
            skills[skill.name] = skill
        return skills

    def load(self, path: Path) -> LoadedSkill:
        instructions_path = path / "SKILL.md"
        handler_path = path / "skill.py"
        if not instructions_path.exists():
            raise SkillLoadError(f"Missing SKILL.md for skill: {path.name}")
        if not handler_path.exists():
            raise SkillLoadError(f"Missing skill.py for skill: {path.name}")
        module_name = f"agent_skill_{path.name}_{abs(hash(path))}"
        spec = importlib.util.spec_from_file_location(module_name, handler_path)
        if spec is None or spec.loader is None:
            raise SkillLoadError(f"Cannot import skill handler: {handler_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metadata = getattr(module, "metadata", None)
        runner = getattr(module, "run", None)
        if not isinstance(metadata, dict):
            raise SkillLoadError(f"Skill metadata must be a dict: {path.name}")
        if not callable(runner) or not inspect.iscoroutinefunction(runner):
            raise SkillLoadError(f"Skill run must be an async function: {path.name}")
        name = str(metadata.get("name") or path.name)
        description = str(metadata.get("description") or "")
        return LoadedSkill(
            name=name,
            description=description,
            instructions=instructions_path.read_text(encoding="utf-8"),
            metadata=metadata,
            run=runner,
        )
