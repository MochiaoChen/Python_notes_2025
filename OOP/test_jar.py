import pytest
from Python_notes_2025.OOP.jar import Jar

class TestJarInit:
    """Test Jar initialization"""
    
    def test_default_capacity(self):
        jar = Jar()
        assert jar.capacity == 12
        assert jar.size == 0
    
    def test_custom_capacity(self):
        jar = Jar(capacity=20)
        assert jar.capacity == 20
        assert jar.size == 0
    
    def test_zero_capacity(self):
        jar = Jar(capacity=0)
        assert jar.capacity == 0
        assert jar.size == 0
    
    def test_negative_capacity_raises_error(self):
        with pytest.raises(ValueError, match="Capacity must be non-negative"):
            Jar(capacity=-1)


class TestJarStr:
    """Test string representation"""
    
    def test_empty_jar(self):
        jar = Jar()
        assert str(jar) == ""
    
    def test_jar_with_cookies(self):
        jar = Jar()
        jar.deposit(3)
        assert str(jar) == "🍪🍪🍪"
    
    def test_jar_full(self):
        jar = Jar(capacity=5)
        jar.deposit(5)
        assert str(jar) == "🍪🍪🍪🍪🍪"


class TestJarDeposit:
    """Test deposit functionality"""
    
    def test_deposit_valid_amount(self):
        jar = Jar()
        jar.deposit(5)
        assert jar.size == 5
    
    def test_deposit_multiple_times(self):
        jar = Jar()
        jar.deposit(3)
        jar.deposit(2)
        assert jar.size == 5
    
    def test_deposit_zero(self):
        jar = Jar()
        jar.deposit(0)
        assert jar.size == 0
    
    def test_deposit_to_full_capacity(self):
        jar = Jar(capacity=10)
        jar.deposit(10)
        assert jar.size == 10
    
    def test_deposit_exceeds_capacity(self):
        jar = Jar(capacity=10)
        jar.deposit(5)
        with pytest.raises(ValueError, match="Exceeds jar capacity"):
            jar.deposit(6)
    
    def test_deposit_negative_amount(self):
        jar = Jar()
        with pytest.raises(ValueError, match="Cannot deposit negative amount"):
            jar.deposit(-1)


class TestJarWithdraw:
    """Test withdraw functionality"""
    
    def test_withdraw_valid_amount(self):
        jar = Jar()
        jar.deposit(10)
        jar.withdraw(5)
        assert jar.size == 5
    
    def test_withdraw_multiple_times(self):
        jar = Jar()
        jar.deposit(10)
        jar.withdraw(3)
        jar.withdraw(2)
        assert jar.size == 5
    
    def test_withdraw_all(self):
        jar = Jar()
        jar.deposit(5)
        jar.withdraw(5)
        assert jar.size == 0
    
    def test_withdraw_zero(self):
        jar = Jar()
        jar.deposit(5)
        jar.withdraw(0)
        assert jar.size == 5
    
    def test_withdraw_more_than_available(self):
        jar = Jar()
        jar.deposit(5)
        with pytest.raises(ValueError, match="Not enough cookies in jar"):
            jar.withdraw(6)
    
    def test_withdraw_from_empty_jar(self):
        jar = Jar()
        with pytest.raises(ValueError, match="Not enough cookies in jar"):
            jar.withdraw(1)
    
    def test_withdraw_negative_amount(self):
        jar = Jar()
        jar.deposit(5)
        with pytest.raises(ValueError, match="Cannot withdraw negative amount"):
            jar.withdraw(-1)


class TestJarProperties:
    """Test capacity and size properties"""
    
    def test_capacity_property(self):
        jar = Jar(capacity=15)
        assert jar.capacity == 15
    
    def test_size_property(self):
        jar = Jar()
        assert jar.size == 0
        jar.deposit(7)
        assert jar.size == 7
        jar.withdraw(3)
        assert jar.size == 4


class TestJarIntegration:
    """Test combined operations"""
    
    def test_deposit_and_withdraw_sequence(self):
        jar = Jar(capacity=20)
        jar.deposit(10)
        jar.withdraw(5)
        jar.deposit(8)
        jar.withdraw(3)
        assert jar.size == 10
    
    def test_fill_and_empty_jar(self):
        jar = Jar(capacity=5)
        jar.deposit(5)
        assert str(jar) == "🍪🍪🍪🍪🍪"
        jar.withdraw(5)
        assert str(jar) == ""