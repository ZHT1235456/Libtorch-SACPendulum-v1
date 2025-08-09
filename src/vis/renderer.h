#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class PendulumRenderer {
public:
    PendulumRenderer(int width=500, int height=500, int radius=180)
    : w_(width), h_(height), L_(radius) {
        canvas_ = cv::Mat::zeros(h_, w_, CV_8UC3);
        origin_ = {w_/2, h_/2 + 120}; // 把枢轴稍微放低一点
        fontFace_ = cv::FONT_HERSHEY_SIMPLEX;
    }

    // theta: 弧度（0 表示竖直向上）
    // action: 当前动作（力矩）
    // reward: 当前奖励
    void render(double theta, double action, double reward, bool show_text=true) {
        canvas_.setTo(cv::Scalar(30,30,30));

        // 画地面
        cv::line(canvas_, {0, origin_.y + 2}, {w_, origin_.y + 2}, cv::Scalar(70,70,70), 2);

        // 枢轴
        cv::circle(canvas_, origin_, 6, cv::Scalar(200,200,200), -1, cv::LINE_AA);

        // 摆端点
        const double x = origin_.x + L_ * std::sin(theta);  // 注意：图里 x 用 sin(theta)
        const double y = origin_.y - L_ * std::cos(theta);
        cv::Point tip{(int)std::round(x), (int)std::round(y)};

        // 杆
        cv::line(canvas_, origin_, tip, cv::Scalar(60,180,255), 6, cv::LINE_AA);

        // 端点小球
        cv::circle(canvas_, tip, 10, cv::Scalar(255,180,60), -1, cv::LINE_AA);

        if (show_text) {
            char buf1[128], buf2[128];
            std::snprintf(buf1, sizeof(buf1), "theta=%.3f rad (%.1f deg)", theta, theta*180.0/M_PI);
            std::snprintf(buf2, sizeof(buf2), "action=%.2f  reward=%.2f", action, reward);
            cv::putText(canvas_, buf1, {20,30}, fontFace_, 0.6, cv::Scalar(230,230,230), 1, cv::LINE_AA);
            cv::putText(canvas_, buf2, {20,55}, fontFace_, 0.6, cv::Scalar(230,230,230), 1, cv::LINE_AA);

            cv::putText(canvas_, "UP is theta=0 (goal)", {20, 80}, fontFace_, 0.5, cv::Scalar(150,200,150), 1, cv::LINE_AA);
        }

        cv::imshow("Pendulum", canvas_);
        // 1ms 非阻塞；按 q 关闭窗口
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q') {
            // 用户可按 q 关闭窗口；不抛异常，静默
            closed_ = true;
            cv::destroyWindow("Pendulum");
        }
    }

    bool is_closed() const { return closed_; }

private:
    int w_, h_, L_;
    cv::Mat canvas_;
    cv::Point origin_;
    int fontFace_;
    bool closed_ = false;
};
