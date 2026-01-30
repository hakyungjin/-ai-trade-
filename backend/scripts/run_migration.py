#!/usr/bin/env python3
"""
Supabase 마이그레이션 스크립트
사용법: python scripts/run_migration.py
"""

import subprocess
import os
import sys
from dotenv import load_dotenv

# .env.production 로드
load_dotenv('.env.production')

def run_migration():
    """Alembic 마이그레이션 실행"""
    
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("❌ DATABASE_URL이 설정되지 않았습니다.")
        print("   .env.production 파일을 확인하세요.")
        sys.exit(1)
    
    print(f"🔗 데이터베이스 연결: {database_url.split('@')[1] if '@' in database_url else 'hidden'}")
    print("⏳ 마이그레이션 실행 중...")
    
    # Alembic 마이그레이션 실행
    result = subprocess.run(
        ['alembic', 'upgrade', 'head'],
        env={**os.environ, 'DATABASE_URL': database_url},
        capture_output=False
    )
    
    if result.returncode == 0:
        print("✅ 마이그레이션 완료!")
        print("📊 테이블 생성됨: coins, stocks, training_data, ...")
        return 0
    else:
        print("❌ 마이그레이션 실패")
        return 1

if __name__ == '__main__':
    sys.exit(run_migration())
