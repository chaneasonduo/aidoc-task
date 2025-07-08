import shutil
import tempfile
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from appserver.service.new_review_service import review_document_with_chain_of_thought

router = APIRouter()

@router.post("/review")
async def review_document(
    file: UploadFile = File(...),
    review_points: List[str] = Form(...)
):
    filename = file.filename or ""
    if not (filename.endswith('.docx') or filename.endswith('.md')):
        raise HTTPException(status_code=400, detail="仅支持 docx 或 md 文件")
    with tempfile.NamedTemporaryFile(delete=False, suffix=filename[-5:]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        results = review_document_with_chain_of_thought(tmp_path, review_points)
    finally:
        try:
            import os
            os.remove(tmp_path)
        except Exception:
            pass
    return results
