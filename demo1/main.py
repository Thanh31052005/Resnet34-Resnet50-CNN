import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.api.predict import router as predict_router

app = FastAPI(title="ResNet Comparison Project")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app.include_router(predict_router)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    context_data = {
        "request": request,
        "title": "Real-time ResNet-34 vs ResNet-50"
    }
    
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context=context_data
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)