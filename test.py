"""
Test script to verify PostgreSQL and Gemini API connectivity
"""

import os
from dotenv import load_dotenv

def test_env_variables():
    """Test if environment variables are set"""
    print("🔍 Testing environment variables...")
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ GOOGLE_API_KEY not configured")
        print("   Please edit .env and add your Gemini API key")
        return False
    
    print("✅ GOOGLE_API_KEY is set")
    return True


def test_postgres_connection():
    """Test PostgreSQL connection"""
    print("\n🔍 Testing PostgreSQL connection...")
    
    try:
        import psycopg
        DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres"

        
        with psycopg.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print(f"✅ Connected to PostgreSQL")
                print(f"   Version: {version[:50]}...")
                return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("   Make sure Docker container is running:")
        print("   docker-compose up -d")
        return False


def test_gemini_api():
    """Test Gemini API connectivity"""
    print("\n🔍 Testing Gemini API...")
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )
        
        response = llm.invoke("Say 'Hello' in one word.")
        print(f"✅ Gemini API is working")
        print(f"   Response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        print("   Check your GOOGLE_API_KEY in .env")
        return False


def test_langgraph():
    """Test LangGraph components"""
    print("\n🔍 Testing LangGraph components...")
    
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore
        
        DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

        
        # Test PostgresSaver (STM)
        with PostgresSaver.from_conn_string(DB_URI) as saver:
            saver.setup()
            print("✅ PostgresSaver (STM) initialized")
        
        # Test PostgresStore (LTM)
        with PostgresStore.from_conn_string(DB_URI) as store:
            store.setup()
            print("✅ PostgresStore (LTM) initialized")
        
        return True
    except Exception as e:
        print(f"❌ LangGraph test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("🧪 TERMINAL CHATBOT TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Environment Variables", test_env_variables()))
    results.append(("PostgreSQL Connection", test_postgres_connection()))
    results.append(("Gemini API", test_gemini_api()))
    results.append(("LangGraph Components", test_langgraph()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to run the chatbot.")
        print("   Run: python terminal_chatbot.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("   Check README.md for troubleshooting steps.")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted")
        exit(1)