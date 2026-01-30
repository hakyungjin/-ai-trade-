#!/usr/bin/env python3
"""
Supabase 테이블 자동 생성 스크립트
사용법: python scripts/run_supabase_init.py
"""

import os
import sys
from dotenv import load_dotenv

# .env.production 로드
load_dotenv('.env.production')

def init_supabase():
    """Supabase에 테이블 생성"""
    
    try:
        import psycopg2
    except ImportError:
        print("❌ psycopg2가 설치되지 않았습니다.")
        print("   설치: pip install psycopg2-binary")
        sys.exit(1)
    
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("❌ DATABASE_URL이 설정되지 않았습니다.")
        print("   .env.production 파일을 확인하세요.")
        sys.exit(1)
    
    # URL 파싱 (postgresql+asyncpg://... → postgresql://...)
    db_url = database_url.replace('postgresql+asyncpg://', 'postgresql://')
    
    print(f"🔗 Supabase 연결 중...")
    print(f"   호스트: {db_url.split('@')[1].split(':')[0] if '@' in db_url else 'hidden'}")
    
    try:
        # psycopg2 직접 연결 (asyncpg 문제 우회)
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        print("✅ Supabase 연결 성공!")
        print("⏳ 테이블 생성 중...\n")
        
        # SQL 파일 읽기
        sql_file = os.path.join(os.path.dirname(__file__), 'init_supabase.sql')
        
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # SQL 실행 (한 줄씩)
        statements = sql_content.split(';')
        table_count = 0
        
        for statement in statements:
            statement = statement.strip()
            if not statement or statement.startswith('--'):
                continue
            
            try:
                cursor.execute(statement)
                if 'CREATE TABLE' in statement:
                    table_name = statement.split('CREATE TABLE IF NOT EXISTS')[1].split('(')[0].strip()
                    print(f"   ✅ {table_name} 테이블 생성")
                    table_count += 1
                elif 'CREATE INDEX' in statement:
                    print(f"   ✅ 인덱스 생성")
            except Exception as e:
                print(f"   ⚠️  경고: {str(e)[:100]}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"\n✅ 성공!")
        print(f"📊 생성된 테이블: {table_count}개")
        print(f"🚀 이제 백엔드를 배포할 수 있습니다!")
        return 0
        
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        print("\n💡 해결책:")
        print("   1. Supabase 대시보드에서 .env.production의 DATABASE_URL 확인")
        print("   2. 또는 Supabase SQL 에디터에서 init_supabase.sql 내용 복사해서 실행")
        return 1

if __name__ == '__main__':
    sys.exit(init_supabase())
