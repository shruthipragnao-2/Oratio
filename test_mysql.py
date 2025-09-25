#!/usr/bin/env python3
"""
Test MySQL integration for Oratio Backend
"""

def test_mysql_connection():
    """Test MySQL connection"""
    try:
        from config import DATABASE_URL
        from sqlalchemy import create_engine
        
        print("Testing MySQL connection...")
        print(f"Database URL: {DATABASE_URL}")
        
        # Test connection
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            result = connection.execute("SELECT 1 as test")
            test_value = result.fetchone()[0]
            
        if test_value == 1:
            print("✅ MySQL connection successful!")
            return True
        else:
            print("❌ MySQL connection test failed")
            return False
            
    except Exception as e:
        print(f"❌ MySQL connection error: {e}")
        print("\nPlease make sure:")
        print("1. MySQL server is running")
        print("2. Database 'oratio' exists")
        print("3. MySQL credentials are correct")
        return False

def test_database_setup():
    """Test database setup"""
    try:
        from main import init_database
        print("\nTesting database setup...")
        init_database()
        print("✅ Database setup successful!")
        return True
    except Exception as e:
        print(f"❌ Database setup error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Oratio MySQL Integration...")
    print("=" * 40)
    
    # Test connection
    connection_ok = test_mysql_connection()
    
    if connection_ok:
        # Test database setup
        setup_ok = test_database_setup()
        
        if setup_ok:
            print("=" * 40)
            print("✅ All MySQL tests passed!")
        else:
            print("=" * 40)
            print("❌ Database setup failed")
    else:
        print("=" * 40)
        print("❌ MySQL connection failed")
