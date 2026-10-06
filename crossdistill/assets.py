"""Resolve teacher weights separately from the tokenizer saved by the experts."""
import hashlib
import json
from pathlib import Path

from opd.local_tokenizer import has_tokenizer_payload, load_local_tokenizer, tokenizer_assets


def tokenizer_signature(tok):
    value = {'vocab': tok.get_vocab(), 'template': tok.chat_template, 'eos': tok.eos_token_id,
             'bos': tok.bos_token_id, 'pad': tok.pad_token_id,
             'backend': tok.backend_tokenizer.to_str()}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def resolve_teacher_tokenizer(teachers, base, teacher_base, explicit=None):
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    candidates = [t['adapter'] for t in teachers]
    candidates += [base['model'].get('tokenizer'), teacher_base]
    for candidate in candidates:
        if candidate and has_tokenizer_payload(candidate):
            return str(Path(candidate).expanduser().resolve())
    raise ValueError('No tokenizer payload found in the five dimension finals, source model.tokenizer '
                     f'or teacher base. Checked: {candidates}. Supply --teacher-tokenizer with the '
                     'original teacher tokenizer directory; the hierarchy root is not a tokenizer.')


def student_tokenizers(models, overrides):
    result = dict(models)
    seen = set()
    for item in overrides:
        name, separator, path = item.partition('=')
        if not separator or name not in models or not path or name in seen:
            raise ValueError('--student-tokenizer requires a unique selected STUDENT=/absolute/path')
        result[name] = str(Path(path).expanduser().resolve())
        seen.add(name)
    return result


def check_tokenizers(paths, teachers):
    loaded, errors, report = {}, [], {}
    for name, path in paths.items():
        report[name] = {'path': path, 'assets': tokenizer_assets(path)}
        try:
            loaded[name] = load_local_tokenizer(path, role=name)
            report[name]['signature'] = tokenizer_signature(loaded[name])
        except (OSError, ValueError) as exc:
            report[name]['error'] = str(exc)
            errors.append(str(exc))
    # Tokenizer files are optional in LoRA exports, but every saved snapshot
    # must agree with the selected teacher tokenizer when present.
    if 'teacher' in loaded:
        for teacher in teachers:
            path = teacher['adapter']
            if not has_tokenizer_payload(path) or Path(path).resolve() == Path(paths['teacher']).resolve():
                continue
            try:
                other = load_local_tokenizer(path, role=f"expert_{teacher['id']}")
                if tokenizer_signature(other) != tokenizer_signature(loaded['teacher']):
                    raise ValueError(f"Teacher tokenizer differs from expert {teacher['id']} snapshot: {path}")
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
    return loaded, report, errors
