from fastapi import APIRouter, HTTPException
from app.models import UserCreate, UserLogin, Token, hash_password, verify_password, create_token, generate_id
from app.database import get_db

router = APIRouter()

@router.post("/signup", response_model=Token)
def signup(user: UserCreate):
    conn = get_db()
    cursor = conn.cursor()
    
    # Check existing
    cursor.execute("SELECT id FROM users WHERE email = ? OR username = ?", (user.email, user.username))
    existing = cursor.fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="Email or username already taken")
    
    user_id = generate_id()
    hashed = hash_password(user.password)
    
    cursor.execute(
        "INSERT INTO users (id, email, username, hashed_password) VALUES (?, ?, ?, ?)",
        (user_id, user.email, user.username, hashed)
    )
    conn.commit()
    conn.close()
    
    token = create_token(user_id, user.username)
    return Token(access_token=token, token_type="bearer", user_id=user_id, username=user.username)

@router.post("/login", response_model=Token)
def login(user: UserLogin):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE email = ?", (user.email,))
    db_user = cursor.fetchone()
    conn.close()
    
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(db_user["id"], db_user["username"])
    return Token(access_token=token, token_type="bearer", user_id=db_user["id"], username=db_user["username"])

@router.get("/me")
def me(authorization: str = None):
    from app.dependencies import get_current_user
    from fastapi import Header
    return {"message": "Use Authorization header"}
