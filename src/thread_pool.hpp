#ifndef THREAD_POOL_HPP_
#define THREAD_POOL_HPP_

#include <thread>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <tuple>
#include <array>
#include <queue>
#include <memory>

namespace cuda_experimental{

//F is function signature to be called by threads in pool
//motivation is to use pool for specific tasks like multithreaded memcpy where signature will be known beforehand
//and minimize overhead e.g. additional allocation
template<std::size_t, std::size_t, typename> class thread_pool;

template<std::size_t ThreadsN, std::size_t TasksN, typename R, typename...Args>
class thread_pool<ThreadsN, TasksN, R(Args...)>{
    using f_type = R(Args...);
    using f_pointer_type = R(*)(Args...);

    class task{
        std::tuple<f_pointer_type, Args...> closure_;
        bool is_complete{false};
        std::condition_variable is_complete_condition;
        std::mutex is_complete_guard;
    public:
        task() = default;
        task(task&&) = default;
        task& operator=(task&&) = default;
        template<typename...Us>
        task(f_type f_, Us...args_):
            closure_{f_, std::forward<Us>(args_)...}
        {}
        void wait(){
            std::unique_lock<std::mutex> lock{is_complete_guard};
            is_complete_condition.wait(lock,[this](){return is_complete;});
        }
    };
    std::mutex tasks_guard;
    std::condition_variable is_empty_tasks_condition;
    std::queue<std::unique_ptr<task>> tasks;
    //std::size_t end{0};
    //std::array<task, TasksN> tasks{};

    void worker_loop(){
        {
            std::unique_lock<std::mutex> lock{tasks_guard};
            is_empty_tasks_condition.wait(lock, [this](){return !tasks.empty();});

        }

    }

public:
    thread_pool() = default;
    //auto size()const{return end;}
    auto size()const{return tasks.size();}
    template<typename...Us>
    auto push(f_type f_, Us...args_){
        {
            std::unique_lock<std::mutex> lock{tasks_guard};
            tasks.push(std::make_unique<task>(f_, std::forward<Us>(args_)...));
            //tasks[end] = std::move(task{f_, std::forward<Us>(args_)...});
            //++end;
        }
    }
};



template<typename T>
class resource_pool
{
    using resource_type = T;


};

template<typename T>
class resource
{
    using resource_type = T;
    std::unique_ptr<resource_type> impl_;
    std::size_t use_count_{0};

    class shared_resource{
        resource* res_;
    public:
        ~shared_resource(){}
        shared_resource(const resource& res__):
            res_{res__}
        {
            ++res_->use_count_;
        }
        shared_resource(const shared_resource& other):
            res_{other.res_}
        {
            ++res_->use_count_;
        }
        shared_resource(shared_resource&& other):
            res_{other.res_}
        {
            ++res_->use_count_;
        }
        shared_resource& operator=(const shared_resource& other)
        {
            if (res_->use_count_ == 1){

            }
            res_ = other.res_;
            ++res_->use_count_;
        }
        auto get()const{return res_->impl_;}
    };

    void return_to_pool(){

    }

public:
    resource():
        impl_{std::make_unique<resource_type>()}
    {}
    std::atomic<>

};



}   //end of namespace cuda_experimental


#endif