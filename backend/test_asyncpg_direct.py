import asyncio
import asyncpg

async def test_connection():
    try:
        # Direct asyncpg connection - force IPv4
        import socket
        
        # Resolve to IPv4 address explicitly
        addr_info = socket.getaddrinfo(
            'db.vmiinfjxpnoevsehhzey.supabase.co',
            6543,
            socket.AF_INET,  # Force IPv4
            socket.SOCK_STREAM
        )
        
        if not addr_info:
            print("❌ No IPv4 address found for Supabase host")
            return
            
        ipv4_addr = addr_info[0][4][0]
        print(f"🔗 Resolved IPv4 address: {ipv4_addr}")
        
        conn = await asyncpg.connect(
            host=ipv4_addr,
            port=6543,
            user='postgres',
            password='ha89498912@@',
            database='postgres',
            ssl=False,
            server_settings={'application_name': 'alembic'}
        )
        
        result = await conn.fetchval('SELECT 1;')
        print(f"✅ Connection successful! Result: {result}")
        await conn.close()
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())
