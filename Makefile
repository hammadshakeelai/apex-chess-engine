CXX = g++
CXXFLAGS = -O3 -std=c++20 -mavx2 -mbmi2 -Wall -Wextra -static -flto -I src/cpp
SRC = $(wildcard src/cpp/*.cpp)
OBJ = $(patsubst src/cpp/%.cpp, bin/%.o, $(SRC))
TARGET = bin/apex_engine.exe

all: $(TARGET)

$(TARGET): $(OBJ)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@
	@echo [OK] Built $(TARGET) successfully!

bin/%.o: src/cpp/%.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf bin/*.o $(TARGET)
