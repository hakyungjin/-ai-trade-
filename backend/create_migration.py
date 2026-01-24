#!/usr/bin/env python3
"""
마이그레이션 파일 생성 스크립트
PowerShell에서 alembic 명령어가 작동하지 않을 때 사용
"""
import subprocess
import sys
import os

# backend 디렉토리로 이동
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# alembic revision --autogenerate 실행
try:
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "revision", "--autogenerate", "-m", "add_coin_monitoring_tables"],
        check=True,
        capture_output=True,
        text=True
    )
    print("✅ 마이그레이션 파일 생성 성공!")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("❌ 마이그레이션 파일 생성 실패:")
    print(e.stderr)
    sys.exit(1)
except FileNotFoundError:
    print("❌ alembic을 찾을 수 없습니다. pip install alembic을 실행하세요.")
    sys.exit(1)


