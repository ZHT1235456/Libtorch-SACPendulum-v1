#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <sstream>

class CSVLogger {
public:
    // 如果 append=false 且文件不存在，会创建并写入 header；
    // 如果 append=true 且文件已存在，则不会重复写 header。
    CSVLogger(const std::string& path,
              const std::vector<std::string>& header,
              bool append = false)
    : path_(path)
    {
        namespace fs = std::filesystem;
        const auto parent = fs::path(path_).parent_path();
        if (!parent.empty()) fs::create_directories(parent);

        const bool file_exists = fs::exists(path_);
        out_.open(path_, append ? std::ios::app : std::ios::out);
        if (!out_.is_open())
            throw std::runtime_error("CSVLogger: cannot open file: " + path_);

        if (!append || !file_exists) {
            write_line_(header);
            out_.flush();
        }
    }

    // 写一行 double（会按默认格式转成字符串）
    void write_row(const std::vector<double>& values) {
        std::vector<std::string> s; s.reserve(values.size());
        for (double v : values) s.push_back(to_string_(v));
        write_line_(s);
    }

    // 写一行 string（你可以自行格式化）
    void write_row(const std::vector<std::string>& values) {
        write_line_(values);
    }

    // 立刻落盘
    void flush() { out_.flush(); }

    ~CSVLogger() { if (out_.is_open()) out_.flush(); }

private:
    std::string path_;
    std::ofstream out_;

    static std::string to_string_(double v) {
        std::ostringstream oss;
        // 简单控制一下小数位，既不太长也不太粗糙
        oss.setf(std::ios::fixed, std::ios::floatfield);
        oss.precision(6);
        oss << v;
        return oss.str();
    }

    // 简单转义：把逗号和双引号做基础处理
    static std::string escape_(const std::string& x) {
        bool need_quote = x.find(',') != std::string::npos || x.find('"') != std::string::npos;
        if (!need_quote) return x;
        std::string y; y.reserve(x.size() + 2);
        y.push_back('"');
        for (char c : x) {
            if (c == '"') y.push_back('"'); // CSV 内部双引号转义为 ""
            y.push_back(c);
        }
        y.push_back('"');
        return y;
    }

    void write_line_(const std::vector<std::string>& cols) {
        for (size_t i = 0; i < cols.size(); ++i) {
            if (i) out_ << ',';
            out_ << escape_(cols[i]);
        }
        out_ << '\n';
    }
};
