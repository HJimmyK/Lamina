/*
        MIT License

        Copyright (c) 2024-2050 Twilight-Dream & With-Sky & HJimmyK

        https://github.com/Twilight-Dream-Of-Magic/
        https://github.com/With-Sky
        https://github.com/HJimmyK

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
*/
/*
        MIT许可证

        版权所有 (c) 2024-2050 Twilight-Dream & With-Sky & HJimmyK

        https://github.com/Twilight-Dream-Of-Magic/
        https://github.com/With-Sky
        https://github.com/HJimmyK

        特此免费授予任何获得本软件及相关文档文件（以下简称“软件”）副本的人，不受限制地处理本软件的权利，
        包括但不限于使用、复制、修改、合并、发布、分发、再许可和/或出售软件的副本，
        以及允许向其提供软件的人这样做，但须符合以下条件：

        上述版权声明和本许可声明应包含在软件的所有副本或主要部分中。

        本软件按“原样”提供，不提供任何形式的明示或暗示担保，
        包括但不限于对适销性、特定用途适用性和非侵权性的担保。
        在任何情况下，作者或版权持有人均不对任何索赔、损害或其他责任负责，
        无论是在合同诉讼、侵权行为或其他方面，由软件或软件的使用或其他交易引起的。
*/

#ifndef HINT_FOR_EASY_BIGINT_HPP
#define HINT_FOR_EASY_BIGINT_HPP

// 包含必要的标准库头文件
#include <iostream>       // 输入输出流
#include <future>         // 异步操作支持
#include <ctime>          // 时间相关函数
#include <climits>        // 基本数据类型大小定义
#include <string>         // 字符串处理
#include <array>          // 固定大小数组
#include <vector>         // 动态数组
#include <type_traits>    // 类型特性判断
#include <random>         // 随机数生成
#include <cassert>        // 断言调试

// Windows 64位系统下的快速乘法宏定义
#if defined(_WIN64)
#include <intrin.h>       // 包含Windows intrinsics函数
#define UMUL128           // 定义UMUL128宏，标记使用_umul128函数
#endif //_WIN64

// GCC编译器下128位整数支持的宏定义
#if defined(__SIZEOF_INT128__)
#define UINT128T          // 定义UINT128T宏，标记使用__uint128_t类型
#endif //__SIZEOF_INT128__

namespace HyperInt
{
    // 生成一个指定bit位数的、所有位均为1的整数（即2^bits - 1）
    // 模板参数T：整数类型
    // 参数bits：指定的bit位数
    // 返回值：类型为T、低bits位全为1的整数
    template <typename T>
    constexpr T all_one(int bits)
    {
        T temp = T(1) << (bits - 1);  
        return temp - 1 + temp;       
    }

    // 计算无符号整数的前导零数量（从最高位开始连续的0的个数）
    // 模板参数IntTy：无符号整数类型（如uint32_t）
    // 参数x：待计算的整数
    // 返回值：前导零的数量
    template <typename IntTy>
    constexpr int hint_clz(IntTy x)
    {
        constexpr uint32_t MASK32 = uint32_t(0xFFFF) << 16;  
        int res = sizeof(IntTy) * CHAR_BIT;  
        if (x & MASK32)  
        {
            res -= 16;
            x >>= 16;    
        }
        if (x & (MASK32 >> 8)) 
        {
            res -= 8;
            x >>= 8;
        }
        if (x & (MASK32 >> 12))  
        {
            res -= 4;
            x >>= 4;
        }
        if (x & (MASK32 >> 14)) 
        {
            res -= 2;
            x >>= 2;
        }
        if (x & (MASK32 >> 15))  
        {
            res -= 1;
            x >>= 1;
        }
        return res - x;  
    }

    // 特化版本：计算uint64_t类型的前导零数量
    constexpr int hint_clz(uint64_t x)
    {
        if (x & (uint64_t(0xFFFFFFFF) << 32))  
        {
            return hint_clz(uint32_t(x >> 32));
        }
       
        return hint_clz(uint32_t(x)) + 32;
    }

    // 计算整数的位长度（即最高有效位的位置+1，0的位长度为0）
    // 模板参数IntTy：整数类型
    // 参数x：待计算的整数
    // 返回值：位长度
    template <typename IntTy>
    constexpr int hint_bit_length(IntTy x)
    {
        if (x == 0)
            return 0;  
        return sizeof(IntTy) * CHAR_BIT - hint_clz(x);
    }

    // 计算整数的以2为底的对数（向下取整，即最高有效位的索引）
    // 模板参数IntTy：整数类型
    // 参数x：待计算的整数（x > 0）
    // 返回值：log2(x)的整数部分
    template <typename IntTy>
    constexpr int hint_log2(IntTy x)
    {
        return (sizeof(IntTy) * CHAR_BIT - 1) - hint_clz(x);
    }

    // 计算uint32_t类型的尾随零数量（从最低位开始连续的0的个数）
    // 参数x：待计算的整数
    // 返回值：尾随零的数量
    constexpr int hint_ctz(uint32_t x)
    {
        int r = 31;  // 初始化为31（最大可能的尾随零）
        x &= (-(int32_t)x);  
        if (x & 0x0000FFFF)  
        {
            r -= 16;
        }
        if (x & 0x00FF00FF)  
        {
            r -= 8;
        }
        if (x & 0x0F0F0F0F)  
        {
            r -= 4;
        }
        if (x & 0x33333333)  
        {
            r -= 2;
        }
        if (x & 0x55555555)  
        {
            r -= 1;
        }
        return r;  
    }

    // 特化版本：计算uint64_t类型的尾随零数量
    constexpr int hint_ctz(uint64_t x)
    {
        if (x & 0xFFFFFFFF)  
            return hint_ctz(uint32_t(x));
        return hint_ctz(uint32_t(x >> 32)) + 32;
    }

    // 快速幂算法：计算m^n（不模任何数）
    // 模板参数T：底数和结果类型，T1：指数类型
    // 参数m：底数，n：指数（n >= 0）
    // 返回值：m的n次幂
    template <typename T, typename T1>
    constexpr T qpow(T m, T1 n)
    {
        T result = 1;  
        while (n > 0)
        {
            if ((n & 1) != 0)  
            {
                result *= m;
            }
            m *= m;  
            n >>= 1; 
        }
        return result;
    }

    // 带模快速幂算法：计算m^n mod mod
    // 模板参数T：底数、模数和结果类型，T1：指数类型
    // 参数m：底数，n：指数（n >= 0），mod：模数
    // 返回值：(m的n次幂)对mod取模的结果
    template <typename T, typename T1>
    constexpr T qpow(T m, T1 n, T mod)
    {
        T result = 1; 
        while (n > 0)
        {
            if ((n & 1) != 0)  
            {
                result *= m;
                result %= mod;
            }
            m *= m;  
            m %= mod;  
            n >>= 1; 
        }
        return result;
    }

    // 计算不大于n的最大2的幂（例如n=5时返回4）
    // 模板参数T：整数类型
    // 参数n：输入整数
    // 返回值：不大于n的最近2的幂
    template <typename T>
    constexpr T int_floor2(T n)
    {
        constexpr int bits = sizeof(n) * CHAR_BIT;  // 该类型的总位数
        for (int i = 1; i < bits; i *= 2)  // 逐步将低位设为1（如1->2->4->...）
        {
            n |= (n >> i);  // 例如n=5(101)，经处理后变为111，右移1加1得100(4)
        }
        return (n >> 1) + 1;  // 得到最大的2的幂
    }

    // 计算不小于n的最小2的幂（例如n=5时返回8）
    // 模板参数T：整数类型
    // 参数n：输入整数
    // 返回值：不小于n的最近2的幂
    template <typename T>
    constexpr T int_ceil2(T n)
    {
        constexpr int bits = sizeof(n) * CHAR_BIT;  // 该类型的总位数
        n--;  // 调整n，处理n本身是2的幂的情况
        for (int i = 1; i < bits; i *= 2)  // 逐步将低位设为1
        {
            n |= (n >> i);  // 例如n=5-1=4(100)，处理后变为111，加1得1000(8)
        }
        return n + 1;  // 得到最小的2的幂
    }

    // 无符号整数加法（带进位）：计算x + y，返回和并更新进位标志
    // 模板参数UintTy：无符号整数类型
    // 参数x, y：加数
    // 参数cf：进位标志（输出，true表示有进位）
    // 返回值：x + y的和
    template <typename UintTy>
    constexpr UintTy add_half(UintTy x, UintTy y, bool& cf)
    {
        x = x + y;  
        cf = (x < y);  
        return x;
    }

    // 无符号整数减法（带借位）：计算x - y，返回差并更新借位标志
    // 模板参数UintTy：无符号整数类型
    // 参数x, y：被减数和减数
    // 参数bf：借位标志（输出，true表示有借位）
    // 返回值：x - y的差
    template <typename UintTy>
    constexpr UintTy sub_half(UintTy x, UintTy y, bool& bf)
    {
        y = x - y;  
        bf = (y > x);  
        return y;
    }

    // 带进位的加法：计算x + y + 进位，返回和并更新进位标志
    // 模板参数UintTy：无符号整数类型
    // 参数x, y：加数
    // 参数cf：进位标志（输入：初始进位；输出：新进位）
    // 返回值：x + y + cf的和
    template <typename UintTy>
    constexpr UintTy add_carry(UintTy x, UintTy y, bool& cf)
    {
        UintTy sum = x + cf;  // 先加初始进位
        cf = (sum < x);  // 记录加进位后的进位
        sum += y;  // 再加y
        cf = cf || (sum < y);  // 若加y后有进位，更新进位标志
        return sum;
    }

    // 带借位的减法：计算x - y - 借位，返回差并更新借位标志
    // 模板参数UintTy：无符号整数类型
    // 参数x, y：被减数和减数
    // 参数bf：借位标志（输入：初始借位；输出：新借位）
    // 返回值：x - y - bf的差
    template <typename UintTy>
    constexpr UintTy sub_borrow(UintTy x, UintTy y, bool& bf)
    {
        UintTy diff = x - bf;  // 先减初始借位
        bf = (diff > x);  // 记录减借位后的借位
        y = diff - y;  // 再减y
        bf = bf || (y > diff);  // 若减y后有借位，更新借位标志
        return y;
    }

    // 扩展欧几里得算法：求解ax + by = gcd(a, b)，返回最大公约数
    // 模板参数IntTy：整数类型
    // 参数a, b：输入整数
    // 参数x, y：输出参数，满足方程的解
    // 返回值：a和b的最大公约数
    template <typename IntTy>
    constexpr IntTy exgcd(IntTy a, IntTy b, IntTy& x, IntTy& y)
    {
        if (b == 0)  // 递归终止条件：b=0时，gcd(a,0)=a，解为x=1,y=0
        {
            x = 1;
            y = 0;
            return a;
        }
        IntTy k = a / b;  // 计算商
        // 递归求解gcd(b, a mod b)，注意交换x和y的位置
        IntTy g = exgcd(b, a - k * b, y, x);
        y -= k * x;  // 调整解y
        return g;
    }

    // 计算模逆：返回n在模mod下的逆元（即n * x ≡ 1 mod mod）
    // 模板参数IntTy：整数类型（要求mod为正数且n与mod互质）
    // 参数n：输入整数，mod：模数
    // 返回值：n的模逆元（在[0, mod)范围内）
    template <typename IntTy>
    constexpr IntTy mod_inv(IntTy n, IntTy mod)
    {
        n %= mod;  // 先对n取模
        IntTy x = 0, y = 0;
        exgcd(n, mod, x, y);  // 利用扩展欧几里得算法求x
        if (x < 0)  // 若x为负，调整到非负范围
        {
            x += mod;
        }
        else if (x >= mod)  // 若x超出mod，调整到范围内
        {
            x -= mod;
        }
        return x;
    }

    // 计算模2^pow下的逆元（牛顿迭代法）
    // 参数n：输入整数（需为奇数，否则无逆元），pow：指数（2^pow为模数）
    // 返回值：n在模2^pow下的逆元
    constexpr uint64_t inv_mod2pow(uint64_t n, int pow)
    {
        const uint64_t mask = all_one<uint64_t>(pow);  // 掩码：低pow位为1
        uint64_t xn = 1, t = n & mask;  // xn初始为1，t为n的低pow位
        while (t != 1)  // 迭代直到t=1（满足xn*n ≡ 1 mod 2^pow）
        {
            xn = (xn * (2 - t));  // 牛顿迭代公式：x_{k+1} = x_k * (2 - n*x_k)
            t = (xn * n) & mask;  // 更新t，检查是否满足条件
        }
        return xn & mask;  // 返回低pow位的逆元
    }

    // 基础算法：64位×64位乘法，结果分为低64位和高64位
    // 参数a, b：乘数
    // 参数low, high：输出参数，分别存储结果的低64位和高64位
    constexpr void mul64x64to128_base(uint64_t a, uint64_t b, uint64_t& low, uint64_t& high)
    {
        uint64_t ah = a >> 32, bh = b >> 32;  // 取a和b的高32位
        a = uint32_t(a), b = uint32_t(b);  // 取a和b的低32位
        // 分块计算：(ah*2^32 + al) * (bh*2^32 + bl) = ah*bh*2^64 + (ah*bl + al*bh)*2^32 + al*bl
        uint64_t r0 = a * b;          // al*bl（低64位中的低32位部分）
        uint64_t r1 = a * bh;         // al*bh（中间项的一部分）
        uint64_t r2 = ah * b;         // ah*bl（中间项的另一部分）
        uint64_t r3 = ah * bh;        // ah*bh（高64位部分）
        r3 += (r1 >> 32) + (r2 >> 32);  // 中间项的进位加到高64位
        r1 = uint32_t(r1), r2 = uint32_t(r2);  // 取r1和r2的低32位
        r1 += r2;  // 合并中间项的低32位
        r1 += (r0 >> 32);  // 加上r0的进位
        high = r3 + (r1 >> 32);  // 高64位结果
        low = (r1 << 32) | uint32_t(r0);  // 低64位结果（中间项低32位左移32位 + r0低32位）
    }

    // 64位×64位乘法的优化实现（根据编译器和平台选择最优方式）
    // 参数a, b：乘数
    // 参数low, high：输出参数，分别存储结果的低64位和高64位
    inline void mul64x64to128(uint64_t a, uint64_t b, uint64_t& low, uint64_t& high)
    {
#if defined(UMUL128)  // Windows 64位平台：使用_umul128 intrinsic函数
#pragma message("使用_umul128计算64位×64位到128位结果")
        unsigned long long lo, hi;
        lo = _umul128(a, b, &hi);  // 调用Windows提供的128位乘法函数
        low = lo, high = hi;
#else
#if defined(UINT128T)  // 支持__uint128_t的编译器（如GCC）
#pragma message("使用__uint128_t计算64位×64位到128位结果")
        __uint128_t x(a);
        x *= b;  // 利用128位整数直接计算
        low = uint64_t(x), high = uint64_t(x >> 64);  // 拆分结果
#else  // 不支持上述两种方式时，使用基础算法
#pragma message("使用基础函数计算64位×64位到128位结果")
        mul64x64to128_base(a, b, low, high);
#endif // UINT128T
#endif // UMUL128
    }

    // 128位整数除以32位整数（将128位被除数分为高64位和低64位）
    // 参数dividend_hi64, dividend_lo64：被除数的高64位和低64位（输入输出，输出为商）
    // 参数divisor：32位除数
    // 返回值：余数
    constexpr uint32_t div128by32(uint64_t& dividend_hi64, uint64_t& dividend_lo64, uint32_t divisor)
    {
        uint32_t quot_hi32 = 0, quot_lo32 = 0;  // 商的高32位和低32位

        // 处理被除数高64位的高32位
        uint64_t dividend = dividend_hi64 >> 32;
        quot_hi32 = dividend / divisor;  // 计算商的高32位
        dividend %= divisor;  // 保留余数

        // 处理被除数高64位的低32位
        dividend = (dividend << 32) | uint32_t(dividend_hi64);
        quot_lo32 = dividend / divisor;  // 计算商的低32位（高64位部分）
        dividend %= divisor;
        dividend_hi64 = (uint64_t(quot_hi32) << 32) | quot_lo32;  // 更新被除数高64位为商的高64位

        // 处理被除数低64位的高32位
        dividend = (dividend << 32) | uint32_t(dividend_lo64 >> 32);
        quot_hi32 = dividend / divisor;  // 计算商的高32位（低64位部分）
        dividend %= divisor;

        // 处理被除数低64位的低32位
        dividend = (dividend << 32) | uint32_t(dividend_lo64);
        quot_lo32 = dividend / divisor;  // 计算商的低32位（低64位部分）
        dividend %= divisor;
        dividend_lo64 = (uint64_t(quot_hi32) << 32) | quot_lo32;  // 更新被除数低64位为商的低64位

        return dividend;  // 返回最终余数
    }

    // 96位整数除以64位整数（商小于2^32）
    // 参数dividend_hi32：被除数的高32位，dividend_lo64：被除数的低64位（输入输出，输出为余数）
    // 参数divisor：64位除数
    // 返回值：32位商
    constexpr uint32_t div96by64to32(uint32_t dividend_hi32, uint64_t& dividend_lo64, uint64_t divisor)
    {
        if (0 == dividend_hi32)  // 若高32位为0，直接用64位除法
        {
            uint32_t quotient = dividend_lo64 / divisor;
            dividend_lo64 %= divisor;
            return quotient;
        }
        // 拆分被除数和除数，进行高位估算
        uint64_t divid2 = (uint64_t(dividend_hi32) << 32) | (dividend_lo64 >> 32);  // 被除数的高64位（96位中的高64）
        uint64_t divis1 = divisor >> 32;  // 除数的高32位
        divisor = uint32_t(divisor);  // 除数的低32位
        uint64_t qhat = divid2 / divis1;  // 估算商的高位
        divid2 %= divis1;  // 保留高位余数
        divid2 = (divid2 << 32) | uint32_t(dividend_lo64);  // 合并余数和被除数低32位
        uint64_t product = qhat * divisor;  // 计算估算商与除数低32位的乘积
        divis1 <<= 32;  // 除数高32位左移32位（恢复为64位的高32位部分）
        // 调整估算商，确保乘积不超过被除数
        if (product > divid2)
        {
            qhat--;
            product -= divisor;
            divid2 += divis1;
            // 再次检查，确保调整正确
            if ((divid2 > divis1) && (product > divid2))
            {
                qhat--;
                product -= divisor;
                divid2 += divis1;
            }
        }
        divid2 -= product;  // 计算最终余数
        dividend_lo64 = divid2;  // 更新余数
        return uint32_t(qhat);  // 返回商（截断为32位）
    }

    // 128位整数除以64位整数（商小于2^64）
    // 参数dividend_hi64：被除数的高64位，dividend_lo64：被除数的低64位（输入输出，输出为余数）
    // 参数divisor：64位除数
    // 返回值：64位商
    constexpr uint64_t div128by64to64(uint64_t dividend_hi64, uint64_t& dividend_lo64, uint64_t divisor)
    {
        int k = 0;
        // 标准化除数：左移使除数的最高位为1，便于估算商
        if (divisor < (uint64_t(1) << 63))
        {
            k = hint_clz(divisor);  // 计算除数的前导零，确定左移位数
            divisor <<= k;  // 标准化除数
            // 同步左移被除数（保持商不变）
            dividend_hi64 = (dividend_hi64 << k) | (dividend_lo64 >> (64 - k));
            dividend_lo64 <<= k;
        }
        // 分两次计算96位除法，合并得到64位商
        uint32_t divid_hi32 = dividend_hi64 >> 32;  // 被除数高64位的高32位
        uint64_t divid_lo64 = (dividend_hi64 << 32) | (dividend_lo64 >> 32);  // 中间96位
        uint64_t quotient = div96by64to32(divid_hi32, divid_lo64, divisor);  // 计算商的高32位

        // 计算商的低32位
        divid_hi32 = divid_lo64 >> 32;  // 中间余数的高32位
        dividend_lo64 = uint32_t(dividend_lo64) | (divid_lo64 << 32);  // 调整被除数低64位
        quotient = (quotient << 32) | div96by64to32(divid_hi32, dividend_lo64, divisor);  // 合并商

        dividend_lo64 >>= k;  // 还原余数（右移k位）
        return quotient;  // 返回64位商
    }

    // 将uint64_t转换为指定位数的十进制字符串（不足补前导零）
    // 参数input：输入的64位无符号整数，digits：指定的字符串长度
    // 返回值：固定长度的十进制字符串
    inline std::string ui64to_string_base10(uint64_t input, uint8_t digits)
    {
        std::string result(digits, '0');  // 初始化字符串为指定长度，填充'0'
        for (uint8_t i = 0; i < digits; i++)
        {
            // 从低位到高位取每一位数字，逆序存入字符串
            result[digits - i - 1] = static_cast<char>(input % 10 + '0');
            input /= 10;  // 移除已处理的低位
        }
        return result;
    }

    // 变换操作命名空间
    namespace Transform
    {
        // 对sum和diff进行变换：sum = sum + diff，diff = sum - diff（原sum和diff）
        // 模板参数T：数据类型
        // 参数sum, diff：输入输出参数，待变换的值
        template <typename T>
        inline void transform2(T& sum, T& diff)
        {
            T temp0 = sum, temp1 = diff; 
            sum = temp0 + temp1; 
            diff = temp0 - temp1;  
        }

        // 数论变换（NTT）相关实现，支持多模数自校验
        namespace NumberTheoreticTransform
        {
            // 预定义的NTT模数及对应的原根（满足模数=2^k * m + 1，原根g的阶为2^k）
            constexpr uint64_t MOD0 = 2485986994308513793, ROOT0 = 5;  // 2^55 * 3 * 23 + 1
            constexpr uint64_t MOD1 = 1945555039024054273, ROOT1 = 5;  // 2^56 * 3^3 + 1
            constexpr uint64_t MOD2 = 4179340454199820289, ROOT2 = 3;  // 2^57 * 29 + 1
            constexpr uint64_t MOD3 = 754974721, ROOT3 = 11;
            constexpr uint64_t MOD4 = 469762049, ROOT4 = 3;
            constexpr uint64_t MOD5 = 3489660929, ROOT5 = 3;
            constexpr uint64_t MOD6 = 3221225473, ROOT6 = 5;

            // 128位无符号整数的内部实现类
            class InternalUInt128
            {
            private:
                uint64_t low;  
                uint64_t high;  

            public:
                // 构造函数：初始化低64位和高64位
                constexpr InternalUInt128(uint64_t l = 0, uint64_t h = 0) : low(l), high(h) {}
                constexpr InternalUInt128(std::pair<uint64_t, uint64_t> p) : low(p.first), high(p.second) {}

                // 加法：当前值 + 另一个InternalUInt128
                constexpr InternalUInt128 operator+(InternalUInt128 rhs) const
                {
                    rhs.low += low; 
                    rhs.high += high + (rhs.low < low);
                    return rhs;
                }

                // 减法：当前值 - 另一个InternalUInt128
                constexpr InternalUInt128 operator-(InternalUInt128 rhs) const
                {
                    rhs.low = low - rhs.low;  
                    rhs.high = high - rhs.high - (rhs.low > low);
                    return rhs;
                }

                // 加法：当前值 + 64位整数
                constexpr InternalUInt128 operator+(uint64_t rhs) const
                {
                    rhs = low + rhs; 
                    return InternalUInt128(rhs, high + (rhs < low));
                }

                // 减法：当前值 - 64位整数
                constexpr InternalUInt128 operator-(uint64_t rhs) const
                {
                    rhs = low - rhs;  
                    return InternalUInt128(rhs, high - (rhs > low));
                }

                // 乘法：当前值（低64位） × 另一个InternalUInt128的低64位
                InternalUInt128 operator*(const InternalUInt128& rhs) const
                {
                    InternalUInt128 res;
                    mul64x64to128(low, rhs.low, res.low, res.high);  
                    return res;
                }

                // 乘法：当前值（低64位） × 64位整数
                InternalUInt128 operator*(uint64_t rhs) const
                {
                    InternalUInt128 res;
                    mul64x64to128(low, rhs, res.low, res.high); 
                    return res;
                }

                // 除法：当前值 / 另一个InternalUInt128（仅用其低64位作为除数）
                constexpr InternalUInt128 operator/(const InternalUInt128& rhs) const
                {
                    return *this / rhs.low;
                }

                // 取余：当前值 % 另一个InternalUInt128（仅用其低64位作为除数）
                constexpr InternalUInt128 operator%(const InternalUInt128& rhs) const
                {
                    return *this % rhs.low;
                }

                // 除法：当前值 / 64位整数（结果为商）
                constexpr InternalUInt128 operator/(uint64_t rhs) const
                {
                    InternalUInt128 quot = *this;
                    quot.selfDivRem(rhs);  // 自更新为商
                    return quot;
                }

                // 取余：当前值 % 64位整数（结果为余数）
                constexpr InternalUInt128 operator%(uint64_t rhs) const
                {
                    InternalUInt128 quot = *this;
                    uint64_t rem = quot.selfDivRem(rhs);  // 获取余数
                    return InternalUInt128(rem);
                }

                // 复合赋值：+= 另一个InternalUInt128
                constexpr InternalUInt128& operator+=(const InternalUInt128& rhs)
                {
                    return *this = *this + rhs;
                }

                // 复合赋值：-= 另一个InternalUInt128
                constexpr InternalUInt128& operator-=(const InternalUInt128& rhs)
                {
                    return *this = *this - rhs;
                }

                // 复合赋值：+= 64位整数
                constexpr InternalUInt128& operator+=(uint64_t rhs)
                {
                    return *this = *this + rhs;
                }

                // 复合赋值：-= 64位整数
                constexpr InternalUInt128& operator-=(uint64_t rhs)
                {
                    return *this = *this - rhs;
                }

                // 复合赋值：*= 另一个InternalUInt128（仅低64位相乘）
                constexpr InternalUInt128& operator*=(const InternalUInt128& rhs)
                {
                    mul64x64to128_base(low, rhs.low, low, high);  
                    return *this;
                }

                // 复合赋值：/= 另一个InternalUInt128
                constexpr InternalUInt128& operator/=(const InternalUInt128& rhs)
                {
                    return *this = *this / rhs;
                }

                // 复合赋值：%= 另一个InternalUInt128
                constexpr InternalUInt128& operator%=(const InternalUInt128& rhs)
                {
                    return *this = *this % rhs;
                }

                // 自更新为商，并返回余数（128位 ÷ 64位）
                constexpr uint64_t selfDivRem(uint64_t divisor)
                {
                    if ((divisor >> 32) == 0)  
                        return div128by32(high, low, uint32_t(divisor));  
                    // 除数为64位
                    uint64_t divid1 = high % divisor, divid0 = low;  // 拆分被除数
                    high /= divisor;  // 高64位商
                    low = div128by64to64(divid1, divid0, divisor);  // 低64位商
                    return divid0;  // 返回余数
                }

                // 静态方法：64位×64位→128位乘法
                static constexpr InternalUInt128 mul64x64(uint64_t a, uint64_t b)
                {
                    InternalUInt128 res;
                    mul64x64to128_base(a, b, res.low, res.high);  
                    return res;
                }

                // 比较：当前值 < 另一个InternalUInt128
                constexpr bool operator<(const InternalUInt128& rhs) const
                {
                    if (high != rhs.high)
                        return high < rhs.high; 
                    return low < rhs.low;  
                }

                // 比较：当前值 == 另一个InternalUInt128
                constexpr bool operator==(const InternalUInt128& rhs) const
                {
                    return high == rhs.high && low == rhs.low;  // 高低64位均相等
                }

                // 左移操作：当前值 << shift位（shift范围0-127）
                constexpr InternalUInt128 operator<<(int shift) const
                {
                    if (shift == 0)
                        return *this;  
                    shift %= 128;  // 取模128，限制移位范围
                    shift = shift < 0 ? shift + 128 : shift;  // 处理负移位（转为右移）
                    if (shift < 64)  
                        return InternalUInt128(low << shift, (high << shift) | (low >> (64 - shift)));
                    // 移位大于等于64位：低64位清零，高64位为低64位左移(shift-64)
                    return InternalUInt128(0, low << (shift - 64));
                }

                // 右移操作：当前值 >> shift位（shift范围0-127）
                constexpr InternalUInt128 operator>>(int shift) const
                {
                    if (shift == 0)
                        return *this; 
                    shift %= 128;  // 取模128，限制移位范围
                    shift = shift < 0 ? shift + 128 : shift;  // 处理负移位（转为左移）
                    if (shift < 64) 
                        return InternalUInt128((low >> shift) | (high << (64 - shift)), high >> shift);
                    // 移位大于等于64位：高64位清零，低64位为高64位右移(shift-64)
                    return InternalUInt128(high >> (shift - 64), 0);
                }

                // 复合赋值：<<= shift位
                constexpr InternalUInt128& operator<<=(int shift)
                {
                    return *this = *this << shift;
                }

                // 复合赋值：>>= shift位
                constexpr InternalUInt128& operator>>=(int shift)
                {
                    return *this = *this >> shift;
                }

                // 获取高64位
                constexpr uint64_t high64() const
                {
                    return high;
                }

                // 获取低64位
                constexpr uint64_t low64() const
                {
                    return low;
                }

                // 转换为uint64_t（仅返回低64位）
                constexpr operator uint64_t() const
                {
                    return low64();
                }

                // 转换为十进制字符串
                std::string toStringBase10() const
                {
                    if (high == 0)  
                        return std::to_string(low);
                    constexpr uint64_t BASE(10000'0000'0000'0000);  // 16位十进制基数（1e16）
                    InternalUInt128 copy(*this);
                    std::string s;
                    // 分三次除以BASE，获取各部分的十进制表示
                    s = ui64to_string_base10(uint64_t(copy.selfDivRem(BASE)), 16) + s;
                    s = ui64to_string_base10(uint64_t(copy.selfDivRem(BASE)), 16) + s;
                    return std::to_string(uint64_t(copy.selfDivRem(BASE))) + s;
                }

                // 打印十进制表示
                void printDec() const
                {
                    std::cout << std::dec << toStringBase10() << '\n';
                }

                // 打印十六进制表示
                void printHex() const
                {
                    std::cout << std::hex << "0x" << high << ' ' << low << std::dec << '\n';
                }
            };

            // 192位无符号整数的内部实现类
            class InternalUInt192
            {
                friend InternalUInt128;  // 允许InternalUInt128访问私有成员

            private:
                uint64_t low;   // 低64位
                uint64_t mid;   // 中64位
                uint64_t high;  // 高64位

            public:
                // 构造函数
                constexpr InternalUInt192() : low(0), mid(0), high(0) {}
                constexpr InternalUInt192(uint64_t low, uint64_t mi = 0, uint64_t high = 0) : low(low), mid(mi), high(high) {}
                // 从128位整数构造（低128位为输入值，高64位为0）
                constexpr InternalUInt192(InternalUInt128 n) : low(n.low64()), mid(n.high64()), high(0) {}

                // 加法：当前值 + 另一个InternalUInt192（带进位传递）
                constexpr InternalUInt192 operator+(InternalUInt192 rhs) const
                {
                    bool cf = false;  // 进位标志
                    rhs.low = add_half(low, rhs.low, cf);  // 加低64位，更新进位
                    rhs.mid = add_carry(mid, rhs.mid, cf);  // 加中64位（带进位），更新进位
                    rhs.high = high + rhs.high + cf;  // 加高64位（带进位）
                    return rhs;
                }

                // 减法：当前值 - 另一个InternalUInt192（带借位传递）
                constexpr InternalUInt192 operator-(InternalUInt192 rhs) const
                {
                    bool bf = false;  // 借位标志
                    rhs.low = sub_half(low, rhs.low, bf);  // 减低64位，更新借位
                    rhs.mid = sub_borrow(mid, rhs.mid, bf);  // 减中64位（带借位），更新借位
                    rhs.high = high - rhs.high - bf;  // 减高64位（带借位）
                    return rhs;
                }

                // 除法：当前值 / 64位整数（结果为商）
                constexpr InternalUInt192 operator/(uint64_t rhs) const
                {
                    InternalUInt192 result(*this);
                    result.selfDivRem(rhs);  // 自更新为商
                    return result;
                }

                // 取余：当前值 % 64位整数（结果为余数）
                constexpr InternalUInt192 operator%(uint64_t rhs) const
                {
                    InternalUInt192 result(*this);
                    return result.selfDivRem(rhs);  // 返回余数
                }

                // 复合赋值：+= 另一个InternalUInt192
                constexpr InternalUInt192& operator+=(const InternalUInt192& rhs)
                {
                    return *this = *this + rhs;
                }

                // 复合赋值：-= 另一个InternalUInt192
                constexpr InternalUInt192& operator-=(const InternalUInt192& rhs)
                {
                    return *this = *this - rhs;
                }

                // 复合赋值：/= 另一个InternalUInt192
                constexpr InternalUInt192& operator/=(const InternalUInt192& rhs)
                {
                    return *this = *this / rhs;
                }

                // 复合赋值：%= 另一个InternalUInt192
                constexpr InternalUInt192& operator%=(const InternalUInt192& rhs)
                {
                    return *this = *this % rhs;
                }

                // 左移操作：当前值 << shift位（shift范围0-191）
                constexpr InternalUInt192 operator<<(int shift) const
                {
                    if (shift == 0)
                        return *this;  
                    shift %= 192;  // 取模192，限制移位范围
                    shift = shift < 0 ? shift + 192 : shift;  // 处理负移位
                    if (shift < 64)  // 移位小于64位：低64位左移，中64位接收低溢出，高64位接收中溢出
                    {
                        return InternalUInt192(
                            low << shift,
                            (mid << shift) | (low >> (64 - shift)),
                            (high << shift) | (mid >> (64 - shift))
                        );
                    }
                    else if (shift < 128)  // 移位64-127位：低64位清零，中64位为低64位左移，高64位接收中溢出
                    {
                        shift -= 64;
                        return InternalUInt192(
                            0,
                            low << shift,
                            (mid << shift) | (low >> (64 - shift))
                        );
                    }
                    // 移位128-191位：低64位和中64位清零，高64位为低64位左移
                    return InternalUInt192(
                        0,
                        0,
                        low << (shift - 128)
                    );
                }

                // 比较：当前值 < 另一个InternalUInt192
                friend constexpr bool operator<(const InternalUInt192& lhs, const InternalUInt192& rhs)
                {
                    if (lhs.high != rhs.high)
                        return lhs.high < rhs.high; 
                    if (lhs.mid != rhs.mid)
                        return lhs.mid < rhs.mid;  
                    return lhs.low < rhs.low;  
                }

                // 比较：当前值 <= 另一个InternalUInt192
                friend constexpr bool operator<=(const InternalUInt192& lhs, const InternalUInt192& rhs)
                {
                    return !(rhs > lhs);  
                }

                // 比较：当前值 > 一个InternalUInt128
                friend constexpr bool operator>(const InternalUInt192& lhs, const InternalUInt128& rhs)
                {
                    return rhs < lhs;  
                }

                // 比较：当前值 >= 另一个InternalUInt192
                friend constexpr bool operator>=(const InternalUInt192& lhs, const InternalUInt192& rhs)
                {
                    return !(lhs < rhs); 
                }

                // 比较：当前值 == 另一个InternalUInt192
                friend constexpr bool operator==(const InternalUInt192& lhs, const InternalUInt192& rhs)
                {
                    return lhs.low == rhs.low && lhs.mid == rhs.mid && lhs.high == rhs.high;  
                }

                // 比较：当前值 != 另一个InternalUInt192
                friend constexpr bool operator!=(const InternalUInt192& lhs, const InternalUInt192& rhs)
                {
                    return !(lhs == rhs);  
                }

                // 静态方法：128位×64位→192位乘法
                static constexpr InternalUInt192 mul128x64(InternalUInt128 a, uint64_t b)
                {
                    auto prod1 = InternalUInt128::mul64x64(b, a.low64());  // b × a的低64位
                    auto prod2 = InternalUInt128::mul64x64(b, a.high64());  // b × a的高64位
                    InternalUInt192 result;
                    result.low = prod1.low64();  // 结果低64位
                    result.mid = prod1.high64() + prod2.low64();  // 结果中64位（带进位）
                    result.high = prod2.high64() + (result.mid < prod1.high64());  // 结果高64位（带进位）
                    return result;
                }

                // 静态方法：64位×64位×64位→192位乘法（先乘前两个，再乘第三个）
                static constexpr InternalUInt192 mul64x64x64(uint64_t a, uint64_t b, uint64_t c)
                {
                    return mul128x64(InternalUInt128::mul64x64(a, b), c);
                }

                // 自更新为商，并返回余数（192位 ÷ 64位）
                constexpr uint64_t selfDivRem(uint64_t divisor)
                {
                    // 处理高64位
                    uint64_t divid1 = high % divisor, divid0 = mid;
                    high /= divisor;  // 高64位商
                    mid = div128by64to64(divid1, divid0, divisor);  // 中64位商

                    // 处理中64位和低64位
                    divid1 = divid0, divid0 = low;
                    low = div128by64to64(divid1, divid0, divisor);  // 低64位商
                    return divid0;  // 返回余数
                }

                // 右移64位（等价于除以2^64）
                constexpr InternalUInt192 rShift64() const
                {
                    return InternalUInt192(mid, high, 0);  // 低64位=中64位，中64位=高64位，高64位=0
                }

                // 转换为uint64_t（仅返回低64位）
                constexpr operator uint64_t() const
                {
                    return low;
                }

                // 转换为十进制字符串
                std::string toStringBase10() const
                {
                    if (high == 0)  // 若高64位为0，用128位转换
                    {
                        return InternalUInt128(mid, low).toStringBase10();
                    }
                    constexpr uint64_t BASE(10000'0000'0000'0000);  // 16位十进制基数
                    InternalUInt192 copy(*this);
                    std::string s;
                    // 分四次除以BASE，获取各部分的十进制表示
                    s = ui64to_string_base10(uint64_t(copy.selfDivRem(BASE)), 16) + s;
                    s = ui64to_string_base10(uint64_t(copy.selfDivRem(BASE)), 16) + s;
                    s = ui64to_string_base10(uint64_t(copy.selfDivRem(BASE)), 16) + s;
                    return std::to_string(uint64_t(copy.selfDivRem(BASE))) + s;
                }

                // 打印十进制表示
                void printDec() const
                {
                    std::cout << std::dec << toStringBase10() << '\n';
                }

                // 打印十六进制表示
                void printHex() const
                {
                    std::cout << std::hex << "0x" << high << ' ' << mid << ' ' << low << std::dec << '\n';
                }
            };

            // 模板函数：获取128位整数的高64位（通用版本）
            template <typename Int128Type>
            constexpr uint64_t high64(const Int128Type& n)
            {
                return n >> 64;  // 右移64位得到高64位
            }

            // 特化版本：获取InternalUInt128的高64位
            constexpr uint64_t high64(const InternalUInt128& n)
            {
                return n.high64();  // 直接调用成员函数
            }

            // 根据编译器是否支持__uint128_t，定义默认的128位整数类型
#ifdef UINT128T
            using UInt128Default = __uint128_t;
#else
            using UInt128Default = InternalUInt128;
#endif // UINT128T

            // Montgomery模运算类（用于模数>2^32的场景，默认基数R=2^64）
            // 模板参数MOD：模数，Int128Type：128位整数类型（默认InternalUInt128）
            template <uint64_t MOD, typename Int128Type = InternalUInt128>
            class MontInt64Lazy
            {
            private:
                // 静态断言：确保模数大于2^32且小于2^62（避免溢出）
                static_assert(MOD > UINT32_MAX, "Montgomery64模数必须大于2^32");
                static_assert(hint_log2(MOD) < 62, "MOD不能大于62位");
                uint64_t data;  // 存储Montgomery形式的数

            public:
                using IntType = uint64_t;  // 基础整数类型

                // 构造函数
                constexpr MontInt64Lazy() : data(0) {}
                // 从普通64位整数构造（转换为Montgomery形式）
                constexpr MontInt64Lazy(uint64_t n) : data(mulMontCompileTime(n, rSquare())) {}

                // 加法：当前值 + 另一个MontInt64Lazy（自动模2*MOD，减少取模次数）
                constexpr MontInt64Lazy operator+(MontInt64Lazy rhs) const
                {
                    rhs.data = data + rhs.data;
                    // 若和大于等于2*MOD，则减2*MOD（保持在[0, 2*MOD)范围内）
                    rhs.data = rhs.data < mod2() ? rhs.data : rhs.data - mod2();
                    return rhs;
                }

                // 减法：当前值 - 另一个MontInt64Lazy（自动调整到非负范围）
                constexpr MontInt64Lazy operator-(MontInt64Lazy rhs) const
                {
                    rhs.data = data - rhs.data;
                    // 若差为负，则加2*MOD（保持在[0, 2*MOD)范围内）
                    rhs.data = rhs.data > data ? rhs.data + mod2() : rhs.data;
                    return rhs;
                }

                // 乘法：当前值 * 另一个MontInt64Lazy（使用运行时优化）
                MontInt64Lazy operator*(MontInt64Lazy rhs) const
                {
                    rhs.data = mulMontRunTimeLazy(data, rhs.data);  // 懒约简（结果可能在[0, 2*MOD)）
                    return rhs;
                }

                // 复合赋值：+= 另一个MontInt64Lazy
                constexpr MontInt64Lazy& operator+=(const MontInt64Lazy& rhs)
                {
                    return *this = *this + rhs;
                }

                // 复合赋值：-= 另一个MontInt64Lazy
                constexpr MontInt64Lazy& operator-=(const MontInt64Lazy& rhs)
                {
                    return *this = *this - rhs;
                }

                // 复合赋值：*= 另一个MontInt64Lazy（使用编译时约简）
                constexpr MontInt64Lazy& operator*=(const MontInt64Lazy& rhs)
                {
                    data = mulMontCompileTime(data, rhs.data);
                    return *this;
                }

                // 标准化：将值调整到[0, MOD)范围内
                constexpr MontInt64Lazy largeNorm2() const
                {
                    MontInt64Lazy res;
                    res.data = data >= mod2() ? data - mod2() : data;
                    return res;
                }

                // 原始加法（不做模调整）
                constexpr MontInt64Lazy rawAdd(MontInt64Lazy rhs) const
                {
                    rhs.data = data + rhs.data;
                    return rhs;
                }

                // 原始减法（加mod2()确保非负）
                constexpr MontInt64Lazy rawSub(MontInt64Lazy rhs) const
                {
                    rhs.data = data - rhs.data + mod2();
                    return rhs;
                }

                // 转换为普通64位整数（从Montgomery形式还原）
                constexpr operator uint64_t() const
                {
                    return toInt(data);
                }

                // 获取模数
                static constexpr uint64_t mod()
                {
                    return MOD;
                }

                // 获取2*模数
                static constexpr uint64_t mod2()
                {
                    return MOD * 2;
                }

                // 计算模数的逆元（模2^64），满足(MOD * modInv) ≡ 1 mod 2^64
                static constexpr uint64_t modInv()
                {
                    constexpr uint64_t mod_inv = inv_mod2pow(mod(), 64);  // 调用模2^pow逆元函数
                    return mod_inv;
                }

                // 计算模数逆元的负数（模2^64），满足(modInvNeg + modInv) ≡ 0 mod 2^64
                static constexpr uint64_t modInvNeg()
                {
                    constexpr uint64_t mod_inv_neg = uint64_t(0 - modInv());  // 等价于~modInv() + 1
                    return mod_inv_neg;
                }

                // 计算R^2 mod MOD，其中R=2^64（Montgomery形式转换的关键参数）
                static constexpr uint64_t rSquare()
                {
                    constexpr Int128Type r = (Int128Type(1) << 64) % Int128Type(mod());  // R mod MOD
                    constexpr uint64_t r2 = uint64_t(qpow(r, 2, Int128Type(mod())));  // R^2 mod MOD
                    return r2;
                }

                // 静态断言：验证模逆是否正确
                static_assert((mod()* modInv()) == 1, "mod_inv不正确");

                // 将普通整数转换为Montgomery形式：x' = x * R^2 mod MOD
                static constexpr uint64_t toMont(uint64_t n)
                {
                    return mulMontCompileTime(n, rSquare());
                }

                // 将Montgomery形式还原为普通整数：x = x' * R^{-1} mod MOD
                static constexpr uint64_t toInt(uint64_t n)
                {
                    return redc(Int128Type(n));  // 调用约简函数
                }

                // 快速懒约简：返回值可能在[0, 2*MOD)范围内（不做最终模调整）
                static uint64_t redcFastLazy(const Int128Type& input)
                {
                    Int128Type n = uint64_t(input) * modInvNeg();  // n = (input低64位) * (-modInv)
                    n = n * mod();  // n = n * MOD
                    n += input;  // n = input + n*MOD
                    return high64(n);  // 高64位即为约简结果（可能 >= MOD）
                }

                // 快速约简：返回值在[0, MOD)范围内
                static uint64_t redcFast(const Int128Type& input)
                {
                    uint64_t n = redcFastLazy(input);
                    return n < mod() ? n : n - mod();  // 若结果 >= MOD则减MOD
                }

                // 约简函数（编译时版本）：将128位输入约简到[0, MOD)
                static constexpr uint64_t redc(const Int128Type& input)
                {
                    Int128Type n = uint64_t(input) * modInvNeg();  // 计算中间值
                    n *= Int128Type(mod());
                    n += input;
                    uint64_t m = high64(n);  // 取高64位
                    return m < mod() ? m : m - mod();  // 调整到[0, MOD)
                }

                // 运行时Montgomery乘法：a * b * R^{-1} mod MOD（结果在[0, MOD)）
                static uint64_t mulMontRunTime(uint64_t a, uint64_t b)
                {
                    return redcFast(Int128Type(a) * b);  // 先乘后约简
                }

                // 运行时懒乘法：结果可能在[0, 2*MOD)（减少一次减法）
                static uint64_t mulMontRunTimeLazy(uint64_t a, uint64_t b)
                {
                    return redcFastLazy(Int128Type(a) * b);
                }

                // 编译时Montgomery乘法：a * b * R^{-1} mod MOD（结果在[0, MOD)）
                static constexpr uint64_t mulMontCompileTime(uint64_t a, uint64_t b)
                {
                    Int128Type prod(a);
                    prod *= Int128Type(b);  // 计算乘积
                    return redc(prod);  // 约简
                }
            };

            // 检查模逆是否正确：验证n * n_inv ≡ 1 mod mod
            // 模板参数IntType：整数类型
            // 参数n, n_inv, mod：待验证的数、其逆元、模数
            // 返回值：是否正确（true为正确）
            template <typename IntType>
            constexpr bool check_inv(uint64_t n, uint64_t n_inv, uint64_t mod)
            {
                n %= mod;
                n_inv %= mod;
                IntType m(n);
                m *= IntType(n_inv);
                m %= IntType(mod);
                return m == IntType(1);
            }

            // 三模数中国剩余定理（CRT）：合并三个模运算结果，得到192位整数
            // 模板参数ModInt1, ModInt2, ModInt3：三个模数对应的MontInt类型
            // 参数n1, n2, n3：三个模运算的结果（分别对应ModInt1, ModInt2, ModInt3）
            // 返回值：合并后的192位整数
            template <typename ModInt1, typename ModInt2, typename ModInt3>
            inline InternalUInt192 crt3(ModInt1 n1, ModInt2 n2, ModInt3 n3)
            {
                // 定义三个模数
                constexpr uint64_t MOD1 = ModInt1::mod(), MOD2 = ModInt2::mod(), MOD3 = ModInt3::mod();
                // 计算模数的乘积（用于最终结果范围）
                constexpr InternalUInt192 MOD123 = InternalUInt192::mul64x64x64(MOD1, MOD2, MOD3);  // MOD1*MOD2*MOD3
                constexpr InternalUInt128 MOD12 = InternalUInt128::mul64x64(MOD1, MOD2);  // MOD1*MOD2
                constexpr InternalUInt128 MOD23 = InternalUInt128::mul64x64(MOD2, MOD3);  // MOD2*MOD3
                constexpr InternalUInt128 MOD13 = InternalUInt128::mul64x64(MOD1, MOD3);  // MOD1*MOD3
                // 计算交叉乘积模当前模数的结果
                constexpr uint64_t MOD23_M1 = InternalUInt128::mul64x64(MOD2 % MOD1, MOD3 % MOD1) % InternalUInt128(MOD1);  // (MOD2*MOD3) mod MOD1
                constexpr uint64_t MOD13_M2 = InternalUInt128::mul64x64(MOD1 % MOD2, MOD3 % MOD2) % InternalUInt128(MOD2);  // (MOD1*MOD3) mod MOD2
                constexpr uint64_t MOD12_M3 = InternalUInt128::mul64x64(MOD1 % MOD3, MOD2 % MOD3) % InternalUInt128(MOD3);  // (MOD1*MOD2) mod MOD3
                // 计算交叉乘积的模逆
                constexpr ModInt1 MOD23_INV1 = mod_inv<int64_t>(MOD23_M1, MOD1);  // (MOD2*MOD3)^-1 mod MOD1
                constexpr ModInt2 MOD13_INV2 = mod_inv<int64_t>(MOD13_M2, MOD2);  // (MOD1*MOD3)^-1 mod MOD2
                constexpr ModInt3 MOD12_INV3 = mod_inv<int64_t>(MOD12_M3, MOD3);                                           // (MOD1*MOD2)^-1 mod MOD3
                static_assert(check_inv<InternalUInt128>(MOD23_INV1, MOD23_M1, MOD1), "INV1 error");
                static_assert(check_inv<InternalUInt128>(MOD13_INV2, MOD13_M2, MOD2), "INV2 error");
                static_assert(check_inv<InternalUInt128>(MOD12_INV3, MOD12_M3, MOD3), "INV3 error");
                n1 = n1 * MOD23_INV1;
                n2 = n2 * MOD13_INV2;
                n3 = n3 * MOD12_INV3;
                InternalUInt192 result = InternalUInt192::mul128x64(MOD23, uint64_t(n1));
                result += InternalUInt192::mul128x64(MOD13, uint64_t(n2));
                result += InternalUInt192::mul128x64(MOD12, uint64_t(n3));
                result = result < MOD123 ? result : result - MOD123;
                return result < MOD123 ? result : result - MOD123;
            }

            namespace SplitRadix
            {
                template <uint64_t ROOT, typename ModIntType>
                inline ModIntType mul_w41(ModIntType n)
                {
                    constexpr ModIntType W_4_1 = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / 4);
                    return n * W_4_1;
                }

                // in: in_out0<4p, in_ou1<4p; in_out2<2p, in_ou3<2p
                // out: in_out0<4p, in_ou1<4p; in_out2<4p, in_ou3<4p
                template <uint64_t ROOT, typename ModIntType>
                inline void dit_butterfly244(ModIntType& in_out0, ModIntType& in_out1, ModIntType& in_out2, ModIntType& in_out3)
                {
                    ModIntType temp0, temp1, temp2, temp3;
                    temp0 = in_out0.largeNorm2();
                    temp1 = in_out1.largeNorm2();
                    temp2 = in_out2 + in_out3;
                    temp3 = in_out2.rawSub(in_out3);
                    temp3 = mul_w41<ROOT>(temp3);
                    in_out0 = temp0.rawAdd(temp2);
                    in_out2 = temp0.rawSub(temp2);
                    in_out1 = temp1.rawAdd(temp3);
                    in_out3 = temp1.rawSub(temp3);
                }

                // in: in_out0<2p, in_ou1<2p; in_out2<2p, in_ou3<2p
                // out: in_out0<2p, in_ou1<2p; in_out2<4p, in_ou3<4p
                template <uint64_t ROOT, typename ModIntType>
                inline void dif_butterfly244(ModIntType& in_out0, ModIntType& in_out1, ModIntType& in_out2, ModIntType& in_out3)
                {
                    ModIntType temp0, temp1, temp2, temp3;
                    temp0 = in_out0.rawAdd(in_out2);
                    temp2 = in_out0 - in_out2;
                    temp1 = in_out1.rawAdd(in_out3);
                    temp3 = in_out1.rawSub(in_out3);
                    temp3 = mul_w41<ROOT>(temp3);
                    in_out0 = temp0.largeNorm2();
                    in_out1 = temp1.largeNorm2();
                    in_out2 = temp2.rawAdd(temp3);
                    in_out3 = temp2.rawSub(temp3);
                }

                // in: in_out0<4p, in_ou1<4p
                // out: in_out0<4p, in_ou1<4p
                template <typename ModIntType>
                inline void dit_butterfly2(ModIntType& in_out0, ModIntType& in_out1, const ModIntType& omega)
                {
                    auto x = in_out0.largeNorm2();
                    auto y = in_out1 * omega;
                    in_out0 = x.rawAdd(y);
                    in_out1 = x.rawSub(y);
                }

                // in: in_out0<2p, in_ou1<2p
                // out: in_out0<2p, in_ou1<2p
                template <typename ModIntType>
                inline void dif_butterfly2(ModIntType& in_out0, ModIntType& in_out1, const ModIntType& omega)
                {
                    auto x = in_out0 + in_out1;
                    auto y = in_out0.rawSub(in_out1);
                    in_out0 = x;
                    in_out1 = y * omega;
                }

                template <size_t MAX_LEN, uint64_t ROOT, typename ModIntType>
                struct NTTShort
                {
                    static constexpr size_t NTT_LEN = MAX_LEN;
                    static constexpr int LOG_LEN = hint_log2(NTT_LEN);
                    struct TableType
                    {
                        std::array<ModIntType, NTT_LEN> omega_table;
                        // Compute in compile time if need.
                        /*constexpr*/ TableType()
                        {
                            for (int omega_log_len = 0; omega_log_len <= LOG_LEN; omega_log_len++)
                            {
                                size_t omega_len = size_t(1) << omega_log_len, omega_count = omega_len / 2;
                                auto it = &omega_table[omega_len / 2];
                                ModIntType root = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / omega_len);
                                ModIntType omega(1);
                                for (size_t i = 0; i < omega_count; i++)
                                {
                                    it[i] = omega;
                                    omega *= root;
                                }
                            }
                        }
                        constexpr ModIntType& operator[](size_t i)
                        {
                            return omega_table[i];
                        }
                        constexpr const ModIntType& operator[](size_t i) const
                        {
                            return omega_table[i];
                        }
                        constexpr const ModIntType* getOmegaIt(size_t len) const
                        {
                            return &omega_table[len / 2];
                        }
                    };

                    static TableType table;

                    static void dit(ModIntType in_out[], size_t len)
                    {
                        len = std::min(NTT_LEN, len);
                        size_t rank = len;
                        if (hint_log2(len) % 2 == 0)
                        {
                            NTTShort<4, ROOT, ModIntType>::dit(in_out, len);
                            for (size_t i = 4; i < len; i += 4)
                            {
                                NTTShort<4, ROOT, ModIntType>::dit(in_out + i);
                            }
                            rank = 16;
                        }
                        else
                        {
                            NTTShort<8, ROOT, ModIntType>::dit(in_out, len);
                            for (size_t i = 8; i < len; i += 8)
                            {
                                NTTShort<8, ROOT, ModIntType>::dit(in_out + i);
                            }
                            rank = 32;
                        }
                        for (; rank <= len; rank *= 4)
                        {
                            size_t gap = rank / 4;
                            auto omega_it = table.getOmegaIt(rank), last_omega_it = table.getOmegaIt(rank / 2);
                            auto it0 = in_out, it1 = in_out + gap, it2 = in_out + gap * 2, it3 = in_out + gap * 3;
                            for (size_t j = 0; j < len; j += rank)
                            {
                                for (size_t i = 0; i < gap; i++)
                                {
                                    auto temp0 = it0[j + i], temp1 = it1[j + i], temp2 = it2[j + i], temp3 = it3[j + i], omega = last_omega_it[i];
                                    dit_butterfly2(temp0, temp1, omega);
                                    dit_butterfly2(temp2, temp3, omega);
                                    dit_butterfly2(temp0, temp2, omega_it[i]);
                                    dit_butterfly2(temp1, temp3, omega_it[gap + i]);
                                    it0[j + i] = temp0, it1[j + i] = temp1, it2[j + i] = temp2, it3[j + i] = temp3;
                                }
                            }
                        }
                    }
                    static void dif(ModIntType in_out[], size_t len)
                    {
                        len = std::min(NTT_LEN, len);
                        size_t rank = len;
                        for (; rank >= 16; rank /= 4)
                        {
                            size_t gap = rank / 4;
                            auto omega_it = table.getOmegaIt(rank), last_omega_it = table.getOmegaIt(rank / 2);
                            auto it0 = in_out, it1 = in_out + gap, it2 = in_out + gap * 2, it3 = in_out + gap * 3;
                            for (size_t j = 0; j < len; j += rank)
                            {
                                for (size_t i = 0; i < gap; i++)
                                {
                                    auto temp0 = it0[j + i], temp1 = it1[j + i], temp2 = it2[j + i], temp3 = it3[j + i], omega = last_omega_it[i];
                                    dif_butterfly2(temp0, temp2, omega_it[i]);
                                    dif_butterfly2(temp1, temp3, omega_it[gap + i]);
                                    dif_butterfly2(temp0, temp1, omega);
                                    dif_butterfly2(temp2, temp3, omega);
                                    it0[j + i] = temp0, it1[j + i] = temp1, it2[j + i] = temp2, it3[j + i] = temp3;
                                }
                            }
                        }
                        if (hint_log2(rank) % 2 == 0)
                        {
                            NTTShort<4, ROOT, ModIntType>::dif(in_out, len);
                            for (size_t i = 4; i < len; i += 4)
                            {
                                NTTShort<4, ROOT, ModIntType>::dif(in_out + i);
                            }
                        }
                        else
                        {
                            NTTShort<8, ROOT, ModIntType>::dif(in_out, len);
                            for (size_t i = 8; i < len; i += 8)
                            {
                                NTTShort<8, ROOT, ModIntType>::dif(in_out + i);
                            }
                        }
                    }
                };
                template <size_t LEN, uint64_t ROOT, typename ModIntType>
                typename NTTShort<LEN, ROOT, ModIntType>::TableType NTTShort<LEN, ROOT, ModIntType>::table;
                template <size_t LEN, uint64_t ROOT, typename ModIntType>
                constexpr size_t NTTShort<LEN, ROOT, ModIntType>::NTT_LEN;
                template <size_t LEN, uint64_t ROOT, typename ModIntType>
                constexpr int NTTShort<LEN, ROOT, ModIntType>::LOG_LEN;

                template <uint64_t ROOT, typename ModIntType>
                struct NTTShort<0, ROOT, ModIntType>
                {
                    static void dit(ModIntType in_out[]) {}
                    static void dif(ModIntType in_out[]) {}
                    static void dit(ModIntType in_out[], size_t len) {}
                    static void dif(ModIntType in_out[], size_t len) {}
                };

                template <uint64_t ROOT, typename ModIntType>
                struct NTTShort<1, ROOT, ModIntType>
                {
                    static void dit(ModIntType in_out[]) {}
                    static void dif(ModIntType in_out[]) {}
                    static void dit(ModIntType in_out[], size_t len) {}
                    static void dif(ModIntType in_out[], size_t len) {}
                };

                template <uint64_t ROOT, typename ModIntType>
                struct NTTShort<2, ROOT, ModIntType>
                {
                    static void dit(ModIntType in_out[])
                    {
                        transform2(in_out[0], in_out[1]);
                    }
                    static void dif(ModIntType in_out[])
                    {
                        transform2(in_out[0], in_out[1]);
                    }
                    static void dit(ModIntType in_out[], size_t len)
                    {
                        if (len < 2)
                        {
                            return;
                        }
                        dit(in_out);
                    }
                    static void dif(ModIntType in_out[], size_t len)
                    {
                        if (len < 2)
                        {
                            return;
                        }
                        dif(in_out);
                    }
                };

                template <uint64_t ROOT, typename ModIntType>
                struct NTTShort<4, ROOT, ModIntType>
                {
                    static void dit(ModIntType in_out[])
                    {
                        auto temp0 = in_out[0];
                        auto temp1 = in_out[1];
                        auto temp2 = in_out[2];
                        auto temp3 = in_out[3];

                        transform2(temp0, temp1);
                        transform2(temp2, temp3);
                        temp3 = mul_w41<ROOT>(temp3);

                        in_out[0] = temp0 + temp2;
                        in_out[1] = temp1 + temp3;
                        in_out[2] = temp0 - temp2;
                        in_out[3] = temp1 - temp3;
                    }
                    static void dif(ModIntType in_out[])
                    {
                        auto temp0 = in_out[0];
                        auto temp1 = in_out[1];
                        auto temp2 = in_out[2];
                        auto temp3 = in_out[3];

                        transform2(temp0, temp2);
                        transform2(temp1, temp3);
                        temp3 = mul_w41<ROOT>(temp3);

                        in_out[0] = temp0 + temp1;
                        in_out[1] = temp0 - temp1;
                        in_out[2] = temp2 + temp3;
                        in_out[3] = temp2 - temp3;
                    }
                    static void dit(ModIntType in_out[], size_t len)
                    {
                        if (len < 4)
                        {
                            NTTShort<2, ROOT, ModIntType>::dit(in_out, len);
                            return;
                        }
                        dit(in_out);
                    }
                    static void dif(ModIntType in_out[], size_t len)
                    {
                        if (len < 4)
                        {
                            NTTShort<2, ROOT, ModIntType>::dif(in_out, len);
                            return;
                        }
                        dif(in_out);
                    }
                };

                template <uint64_t ROOT, typename ModIntType>
                struct NTTShort<8, ROOT, ModIntType>
                {
                    static void dit(ModIntType in_out[])
                    {
                        static constexpr ModIntType w1 = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / 8);
                        static constexpr ModIntType w2 = qpow(w1, 2);
                        static constexpr ModIntType w3 = qpow(w1, 3);
                        auto temp0 = in_out[0];
                        auto temp1 = in_out[1];
                        auto temp2 = in_out[2];
                        auto temp3 = in_out[3];
                        auto temp4 = in_out[4];
                        auto temp5 = in_out[5];
                        auto temp6 = in_out[6];
                        auto temp7 = in_out[7];

                        transform2(temp0, temp1);
                        transform2(temp2, temp3);
                        transform2(temp4, temp5);
                        transform2(temp6, temp7);
                        temp3 = mul_w41<ROOT>(temp3);
                        temp7 = mul_w41<ROOT>(temp7);

                        transform2(temp0, temp2);
                        transform2(temp1, temp3);
                        transform2(temp4, temp6);
                        transform2(temp5, temp7);
                        temp5 = temp5 * w1;
                        temp6 = temp6 * w2;
                        temp7 = temp7 * w3;

                        in_out[0] = temp0 + temp4;
                        in_out[1] = temp1 + temp5;
                        in_out[2] = temp2 + temp6;
                        in_out[3] = temp3 + temp7;
                        in_out[4] = temp0 - temp4;
                        in_out[5] = temp1 - temp5;
                        in_out[6] = temp2 - temp6;
                        in_out[7] = temp3 - temp7;
                    }
                    static void dif(ModIntType in_out[])
                    {
                        static constexpr ModIntType w1 = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / 8);
                        static constexpr ModIntType w2 = qpow(w1, 2);
                        static constexpr ModIntType w3 = qpow(w1, 3);
                        auto temp0 = in_out[0];
                        auto temp1 = in_out[1];
                        auto temp2 = in_out[2];
                        auto temp3 = in_out[3];
                        auto temp4 = in_out[4];
                        auto temp5 = in_out[5];
                        auto temp6 = in_out[6];
                        auto temp7 = in_out[7];

                        transform2(temp0, temp4);
                        transform2(temp1, temp5);
                        transform2(temp2, temp6);
                        transform2(temp3, temp7);
                        temp5 = temp5 * w1;
                        temp6 = temp6 * w2;
                        temp7 = temp7 * w3;

                        transform2(temp0, temp2);
                        transform2(temp1, temp3);
                        transform2(temp4, temp6);
                        transform2(temp5, temp7);
                        temp3 = mul_w41<ROOT>(temp3);
                        temp7 = mul_w41<ROOT>(temp7);

                        in_out[0] = temp0 + temp1;
                        in_out[1] = temp0 - temp1;
                        in_out[2] = temp2 + temp3;
                        in_out[3] = temp2 - temp3;
                        in_out[4] = temp4 + temp5;
                        in_out[5] = temp4 - temp5;
                        in_out[6] = temp6 + temp7;
                        in_out[7] = temp6 - temp7;
                    }
                    static void dit(ModIntType in_out[], size_t len)
                    {
                        if (len < 8)
                        {
                            NTTShort<4, ROOT, ModIntType>::dit(in_out, len);
                            return;
                        }
                        dit(in_out);
                    }
                    static void dif(ModIntType in_out[], size_t len)
                    {
                        if (len < 8)
                        {
                            NTTShort<4, ROOT, ModIntType>::dif(in_out, len);
                            return;
                        }
                        dif(in_out);
                    }
                };

                template <uint64_t MOD, uint64_t ROOT, typename Int128Type = UInt128Default>
                struct NTT
                {
                    static constexpr uint64_t mod()
                    {
                        return MOD;
                    }
                    static constexpr uint64_t root()
                    {
                        return ROOT;
                    }
                    static constexpr uint64_t rootInv()
                    {
                        constexpr uint64_t IROOT = mod_inv<int64_t>(ROOT, MOD);
                        return IROOT;
                    }

                    static_assert(root() < mod(), "ROOT must be smaller than MOD");
                    static_assert(check_inv<Int128Type>(root(), rootInv(), mod()), "IROOT * ROOT % MOD must be 1");
                    static constexpr int MOD_BITS = hint_log2(mod()) + 1;
                    static constexpr int MAX_LOG_LEN = hint_ctz(mod() - 1);

                    static constexpr size_t getMaxLen()
                    {
                        if constexpr (MAX_LOG_LEN < sizeof(size_t) * CHAR_BIT)
                        {
                            return size_t(1) << MAX_LOG_LEN;
                        }
                        return size_t(1) << (sizeof(size_t) * CHAR_BIT - 1);
                    }
                    static constexpr size_t NTT_MAX_LEN = getMaxLen();

                    using INTT = NTT<mod(), rootInv(), Int128Type>;
                    using ModInt64Type = MontInt64Lazy<MOD, Int128Type>;
                    using ModIntType = ModInt64Type;
                    using IntType = typename ModIntType::IntType;

                    static constexpr size_t L2_BYTE = size_t(1) << 20; // 1MB L2 cache size, change this if you know your cache size.
                    static constexpr size_t LONG_THRESHOLD = std::min(L2_BYTE / sizeof(ModIntType), NTT_MAX_LEN);
                    using NTTTemplate = NTTShort<LONG_THRESHOLD, root(), ModIntType>;

                    static void dit244(ModIntType in_out[], size_t ntt_len)
                    {
                        ntt_len = std::min(int_floor2(ntt_len), NTT_MAX_LEN);
                        if (ntt_len <= LONG_THRESHOLD)
                        {
                            NTTTemplate::dit(in_out, ntt_len);
                            return;
                        }
                        size_t quarter_len = ntt_len / 4;
                        dit244(in_out + quarter_len * 3, ntt_len / 4);
                        dit244(in_out + quarter_len * 2, ntt_len / 4);
                        dit244(in_out, ntt_len / 2);
                        const ModIntType unit_omega1 = qpow(ModIntType(root()), (mod() - 1) / ntt_len);
                        const ModIntType unit_omega3 = qpow(unit_omega1, 3);
                        ModIntType omega1(1), omega3(1);
                        auto it0 = in_out, it1 = in_out + quarter_len, it2 = in_out + quarter_len * 2, it3 = in_out + quarter_len * 3;
                        for (size_t i = 0; i < quarter_len; i++)
                        {
                            ModIntType temp0 = it0[i], temp1 = it1[i], temp2 = it2[i] * omega1, temp3 = it3[i] * omega3;
                            dit_butterfly244<ROOT>(temp0, temp1, temp2, temp3);
                            it0[i] = temp0, it1[i] = temp1, it2[i] = temp2, it3[i] = temp3;
                            omega1 = omega1 * unit_omega1;
                            omega3 = omega3 * unit_omega3;
                        }
                    }
                    static void dif244(ModIntType in_out[], size_t ntt_len)
                    {
                        ntt_len = std::min(int_floor2(ntt_len), NTT_MAX_LEN);
                        if (ntt_len <= LONG_THRESHOLD)
                        {
                            NTTTemplate::dif(in_out, ntt_len);
                            return;
                        }
                        size_t quarter_len = ntt_len / 4;
                        const ModIntType unit_omega1 = qpow(ModIntType(root()), (mod() - 1) / ntt_len);
                        const ModIntType unit_omega3 = qpow(unit_omega1, 3);
                        ModIntType omega1(1), omega3(1);
                        auto it0 = in_out, it1 = in_out + quarter_len, it2 = in_out + quarter_len * 2, it3 = in_out + quarter_len * 3;
                        for (size_t i = 0; i < quarter_len; i++)
                        {
                            ModIntType temp0 = it0[i], temp1 = it1[i], temp2 = it2[i], temp3 = it3[i];
                            dif_butterfly244<ROOT>(temp0, temp1, temp2, temp3);
                            it0[i] = temp0, it1[i] = temp1, it2[i] = temp2 * omega1, it3[i] = temp3 * omega3;
                            omega1 = omega1 * unit_omega1;
                            omega3 = omega3 * unit_omega3;
                        }
                        dif244(in_out, ntt_len / 2);
                        dif244(in_out + quarter_len * 3, ntt_len / 4);
                        dif244(in_out + quarter_len * 2, ntt_len / 4);
                    }
                    static void convolution(ModIntType in1[], ModIntType in2[], ModIntType out[], size_t ntt_len, bool normlize = true)
                    {
                        dif244(in1, ntt_len);
                        dif244(in2, ntt_len);
                        if (normlize)
                        {
                            const ModIntType inv_len(qpow(ModIntType(ntt_len), mod() - 2));
                            for (size_t i = 0; i < ntt_len; i++)
                            {
                                out[i] = in1[i] * in2[i] * inv_len;
                            }
                        }
                        else
                        {
                            for (size_t i = 0; i < ntt_len; i++)
                            {
                                out[i] = in1[i] * in2[i];
                            }
                        }
                        INTT::dit244(out, ntt_len);
                    }
                    static void convolutionRecursion(ModIntType in1[], ModIntType in2[], ModIntType out[], size_t ntt_len, bool normlize = true)
                    {
                        if (ntt_len <= LONG_THRESHOLD)
                        {
                            NTTTemplate::dif(in1, ntt_len);
                            if (in1 != in2)
                            {
                                NTTTemplate::dif(in2, ntt_len);
                            }
                            if (normlize)
                            {
                                const ModIntType inv_len(qpow(ModIntType(ntt_len), mod() - 2));
                                for (size_t i = 0; i < ntt_len; i++)
                                {
                                    out[i] = in1[i] * in2[i] * inv_len;
                                }
                            }
                            else
                            {
                                for (size_t i = 0; i < ntt_len; i++)
                                {
                                    out[i] = in1[i] * in2[i];
                                }
                            }
                            INTT::NTTTemplate::dit(out, ntt_len);
                            return;
                        }
                        const size_t quarter_len = ntt_len / 4;
                        ModIntType unit_omega1 = qpow(ModIntType(root()), (mod() - 1) / ntt_len);
                        ModIntType unit_omega3 = qpow(unit_omega1, 3);
                        ModIntType omega1(1), omega3(1);
                        if (in1 != in2)
                        {
                            for (size_t i = 0; i < quarter_len; i++)
                            {
                                ModIntType temp0 = in1[i], temp1 = in1[quarter_len + i], temp2 = in1[quarter_len * 2 + i], temp3 = in1[quarter_len * 3 + i];
                                dif_butterfly244<ROOT>(temp0, temp1, temp2, temp3);
                                in1[i] = temp0, in1[quarter_len + i] = temp1, in1[quarter_len * 2 + i] = temp2 * omega1, in1[quarter_len * 3 + i] = temp3 * omega3;

                                temp0 = in2[i], temp1 = in2[quarter_len + i], temp2 = in2[quarter_len * 2 + i], temp3 = in2[quarter_len * 3 + i];
                                dif_butterfly244<ROOT>(temp0, temp1, temp2, temp3);
                                in2[i] = temp0, in2[quarter_len + i] = temp1, in2[quarter_len * 2 + i] = temp2 * omega1, in2[quarter_len * 3 + i] = temp3 * omega3;

                                omega1 = omega1 * unit_omega1;
                                omega3 = omega3 * unit_omega3;
                            }
                        }
                        else
                        {
                            for (size_t i = 0; i < quarter_len; i++)
                            {
                                ModIntType temp0 = in1[i], temp1 = in1[quarter_len + i], temp2 = in1[quarter_len * 2 + i], temp3 = in1[quarter_len * 3 + i];
                                dif_butterfly244<ROOT>(temp0, temp1, temp2, temp3);
                                in1[i] = temp0, in1[quarter_len + i] = temp1, in1[quarter_len * 2 + i] = temp2 * omega1, in1[quarter_len * 3 + i] = temp3 * omega3;

                                omega1 = omega1 * unit_omega1;
                                omega3 = omega3 * unit_omega3;
                            }
                        }

                        convolutionRecursion(in1, in2, out, ntt_len / 2, false);
                        convolutionRecursion(in1 + quarter_len * 2, in2 + quarter_len * 2, out + quarter_len * 2, ntt_len / 4, false);
                        convolutionRecursion(in1 + quarter_len * 3, in2 + quarter_len * 3, out + quarter_len * 3, ntt_len / 4, false);

                        unit_omega1 = qpow(ModIntType(rootInv()), (mod() - 1) / ntt_len);
                        unit_omega3 = qpow(unit_omega1, 3);
                        if (normlize)
                        {
                            const ModIntType inv_len(qpow(ModIntType(ntt_len), mod() - 2));
                            omega1 = inv_len, omega3 = inv_len;
                            for (size_t i = 0; i < quarter_len; i++)
                            {
                                ModIntType temp0 = out[i] * inv_len, temp1 = out[quarter_len + i] * inv_len, temp2 = out[quarter_len * 2 + i] * omega1, temp3 = out[quarter_len * 3 + i] * omega3;
                                dit_butterfly244<rootInv()>(temp0, temp1, temp2, temp3);
                                out[i] = temp0, out[quarter_len + i] = temp1, out[quarter_len * 2 + i] = temp2, out[quarter_len * 3 + i] = temp3;

                                omega1 = omega1 * unit_omega1;
                                omega3 = omega3 * unit_omega3;
                            }
                        }
                        else
                        {
                            omega1 = 1, omega3 = 1;
                            for (size_t i = 0; i < quarter_len; i++)
                            {
                                ModIntType temp0 = out[i], temp1 = out[quarter_len + i], temp2 = out[quarter_len * 2 + i] * omega1, temp3 = out[quarter_len * 3 + i] * omega3;
                                dit_butterfly244<rootInv()>(temp0, temp1, temp2, temp3);
                                out[i] = temp0, out[quarter_len + i] = temp1, out[quarter_len * 2 + i] = temp2, out[quarter_len * 3 + i] = temp3;

                                omega1 = omega1 * unit_omega1;
                                omega3 = omega3 * unit_omega3;
                            }
                        }
                    }
                };
                template <uint64_t MOD, uint64_t ROOT, typename Int128Type>
                constexpr int NTT<MOD, ROOT, Int128Type>::MOD_BITS;
                template <uint64_t MOD, uint64_t ROOT, typename Int128Type>
                constexpr int NTT<MOD, ROOT, Int128Type>::MAX_LOG_LEN;
                template <uint64_t MOD, uint64_t ROOT, typename Int128Type>
                constexpr size_t NTT<MOD, ROOT, Int128Type>::NTT_MAX_LEN;
            } // namespace SplitRadix

            using NTT0 = SplitRadix::NTT<MOD0, ROOT0>; // using 64bit integer, Montgomery speed up
            using NTT1 = SplitRadix::NTT<MOD1, ROOT1>; // using 64bit integer, Montgomery speed up
            using NTT2 = SplitRadix::NTT<MOD2, ROOT2>; // using 64bit integer, Montgomery speed up
        } // namespace NumberTheoreticTransform
    } // namespace Transform

    namespace Arithmetic
    {
        using Transform::NumberTheoreticTransform::InternalUInt128;

        // 将输入数组中的每个元素左移指定的位数，并将结果存储在输出数组中。
        // 返回值是输入数组 in 最后一个元素经过左移 shift 位后，高位溢出部分的值
        // （即 in[len-1] 左移 shift 位后被截断的高位，具体为 in[len-1] >> (WORD_BITS - shift)）
        // 注意：如果 shift >= WORD_BITS，则返回值将为 0
        template <typename WordTy>
        constexpr WordTy lshift_in_word_half(const WordTy in[], size_t len, WordTy out[], int shift)
        {
            constexpr int WORD_BITS = sizeof(WordTy) * CHAR_BIT;
            assert(shift >= 0 && shift < WORD_BITS);
            if (0 == len)
            {
                return 0;
            }
            if (0 == shift)
            {
                std::copy(in, in + len, out);
                return 0;
            }
            // [n,last] -> [?,n >> shift_rem | last << shift]
            WordTy last = in[len - 1], ret = last;
            const int shift_rem = WORD_BITS - shift;
            size_t i = len - 1;
            while (i > 0)
            {
                i--;
                WordTy n = in[i];
                out[i + 1] = (last << shift) | (n >> shift_rem);
                last = n;
            }
            out[0] = last << shift;
            return ret >> shift_rem;
        }
        // 将输入数组 in 中的每个元素左移指定的位数 shift，并将结果存储在输出数组 out 中。
        // out的长度至少为为len+1，最后一个元素将被设置为高位溢出部分的值
        template <typename WordTy>
        constexpr void lshift_in_word(const WordTy in[], size_t len, WordTy out[], int shift)
        {
            if (0 == len)
            {
                return;
            }
            assert(shift >= 0 && size_t(shift) < sizeof(WordTy) * CHAR_BIT);
            uint64_t last = lshift_in_word_half(in, len, out, shift);
            out[len] = last;
        }

        template <typename WordTy>
        constexpr void rshift_in_word(const WordTy in[], size_t len, WordTy out[], int shift)
        {
            constexpr int WORD_BITS = sizeof(WordTy) * CHAR_BIT;
            if (0 == len)
            {
                return;
            }
            if (0 == shift)
            {
                std::copy(in, in + len, out);
                return;
            }
            assert(shift >= 0 && size_t(shift) < sizeof(WordTy) * CHAR_BIT);
            WordTy last = in[0];
            const int shift_rem = WORD_BITS - shift;
            for (size_t i = 1; i < len; i++)
            {
                WordTy n = in[i];
                out[i - 1] = (last >> shift) | (n << shift_rem);
                last = n;
            }
            out[len - 1] = last >> shift;
        }

        // 去除前导零，返回实际长度，如果数组为空，则返回0
        template <typename T>
        constexpr size_t remove_leading_zeros(const T array[], size_t length)
        {
            if (array == nullptr)
            {
                return 0;
            }
            while (length > 0 && array[length - 1] == 0)
            {
                length--;
            }
            return length;
        }

        size_t get_add_len(size_t l_len, size_t r_len)
        {
            return std::max(l_len, r_len) + 1;
        }

        size_t get_sub_len(size_t l_len, size_t r_len)
        {
            return std::max(l_len, r_len);
        }

        size_t get_mul_len(size_t l_len, size_t r_len)
        {
            if (l_len == 0 || r_len == 0)
            {
                return 0;
            }
            return l_len + r_len;
        }

        size_t get_div_len(size_t l_len, size_t r_len)
        {
            return l_len - r_len + 1;
        }

        // 用于计算两个无符号整数数组的绝对加法和，并处理进位。
        template <typename UintTy>
        constexpr bool abs_add_binary_half(const UintTy a[], size_t len_a, const UintTy b[], size_t len_b, UintTy sum[])
        {
            bool carry = false;
            size_t i = 0, min_len = std::min(len_a, len_b);
            for (; i < min_len; i++)
            {
                sum[i] = add_carry(a[i], b[i], carry);
            }
            for (; i < len_a; i++)
            {
                sum[i] = add_half(a[i], UintTy(carry), carry);
            }
            for (; i < len_b; i++)
            {
                sum[i] = add_half(b[i], UintTy(carry), carry);
            }
            return carry;
        }
        // 用于计算两个无符号整数数组的绝对加法，并将结果存储在另一个数组中，同时处理进位。
        // 注意：sum 的长度至少为 max(len_a,len_b) + 1，最后一位将会被处理为进位
        template <typename UintTy>
        constexpr void abs_add_binary(const UintTy a[], size_t len_a, const UintTy b[], size_t len_b, UintTy sum[])
        {
            bool carry = abs_add_binary_half(a, len_a, b, len_b, sum);
            sum[std::max(len_a, len_b)] = carry;
        }

        // 计算两个无符号整数数组的绝对差，并返回是否发生借位。
        // 注意：diff 的长度至少为 max(len_a,len_b)，最后一位将会被处理为借位
        template <typename UintTy>
        constexpr bool abs_sub_binary(const UintTy a[], size_t len_a, const UintTy b[], size_t len_b, UintTy diff[], bool assign_borow = false)
        {
            bool borrow = false;
            size_t i = 0, min_len = std::min(len_a, len_b);
            for (; i < min_len; i++)
            {
                diff[i] = sub_borrow(a[i], b[i], borrow);
            }
            for (; i < len_a; i++)
            {
                diff[i] = sub_half(a[i], UintTy(borrow), borrow);
            }
            for (; i < len_b; i++)
            {
                diff[i] = sub_half(UintTy(0) - UintTy(borrow), b[i], borrow);
            }
            if (assign_borow)
            {
                diff[i] = UintTy(borrow); // 借位
            }
            return borrow;
        }

        // 用于计算给定无符号整数数组（a）与一个无符号整数（num）之间的绝对差，并将结果存储在另一个数组（diff）中
        template <typename UintTy>
        constexpr bool abs_sub_num_binary(const UintTy a[], size_t len_a, UintTy num, UintTy diff[])
        {
            assert(len_a > 0);
            bool borrow = false;
            size_t i = 1;
            diff[0] = sub_half(a[0], num, borrow);
            for (; i < len_a; i++)
            {
                diff[i] = sub_half(a[i], UintTy(borrow), borrow);
            }
            return borrow;
        }

        // Absolute compare, return 1 if a > b, -1 if a < b, 0 if a == b
        // Return the diffence length if a != b
         
        // 绝对值比较，如果 a > b 返回 1，如果 a < b 返回 -1，如果 a == b 返回 0
        // 如果 a!= b，返回 a 和 b 的差的长度
        template <typename T>
        [[nodiscard]] constexpr auto abs_compare(const T in1[], const T in2[], size_t len)
        {
            struct CompareResult
            {
                size_t diff_len;
                int cmp = 0;
            };
            while (len > 0)
            {
                len--;
                if (in1[len] != in2[len])
                {
                    CompareResult result{ len + 1, 0 };
                    result.cmp = in1[len] > in2[len] ? 1 : -1;
                    return result;
                }
            }
            return CompareResult{ 0, 0 };
        }

        // 绝对值比较，如果 a > b 返回 1，如果 a < b 返回 -1，如果 a == b 返回 0
        template <typename T>
        [[nodiscard]] constexpr int abs_compare(const T in1[], size_t len1, const T in2[], size_t len2)
        {
            if (len1 != len2)
            {
                return len1 > len2 ? 1 : -1;
            }
            return abs_compare(in1, in2, len1).cmp;
        }

        // 用于计算两个二进制无符号整数数组之间的绝对差，并将结果存储在指定的数组（diff）中。
        // 返回值为符号，-1 表示 a < b，0 表示 a == b，1 表示 a > b
        // 注意：diff 的长度至少为 max(len_a,len_b)，最后一位将会被处理为借位
        template <typename UintTy>
        [[nodiscard]] constexpr int abs_difference_binary(const UintTy a[], size_t len1, const UintTy b[], size_t len2, UintTy diff[])
        {
            int sign = 1;
            if (len1 == len2)
            {
                auto cmp = abs_compare(a, b, len1);
                sign = cmp.cmp;
                std::fill(diff + cmp.diff_len, diff + len1, UintTy(0));
                len1 = len2 = cmp.diff_len;
                if (sign < 0)
                {
                    std::swap(a, b);
                }
            }
            else if (len1 < len2)
            {
                std::swap(a, b);
                std::swap(len1, len2);
                sign = -1;
            }
            abs_sub_binary(a, len1, b, len2, diff);
            return sign;
        }
        // 朴素乘法复杂度
        // T = 3.6 * len1 * len2
        inline size_t mul_classic_complexity(size_t len1, size_t len2)
        {
            constexpr double CLASSIC_COMPLEXITY_CONSTANT = 3.6;
            return CLASSIC_COMPLEXITY_CONSTANT * len1 * len2;
        }
        // len1 > len2
        // karatsuba乘法复杂度
        // T = 19 * len1^1.58 * sqrt(len2/len1)
        inline size_t mul_karatsuba_complexity(size_t len1, size_t len2)
        {
            static const double log2_3 = std::log2(3);
            constexpr double KARATSUBA_COMPLEXITY_CONSTANT = 19;
            return KARATSUBA_COMPLEXITY_CONSTANT * std::pow(len1, log2_3) * std::sqrt(double(len2) / len1);
        }
        // 3ntt-crt乘法复杂度
        // ntt_len = int_ceil2(len1 + len2 - 1)
        // T = 40 * ntt_len * log2(ntt_len)
        inline size_t mul_ntt_complexity(size_t len1, size_t len2)
        {
            constexpr int NTT_COMPLEXITY_CONSTANT = 40;
            size_t ntt_len = int_ceil2(len1 + len2 - 1);
            return NTT_COMPLEXITY_CONSTANT * ntt_len * hint_log2(ntt_len);
        }
        // 输入参数：
        //     const uint64_t in[]：输入的无符号64位整数数组    
        //     size_t len：输入数组 in 的长度（元素个数）。
        //     uint64_t out[]：输出数组，长度至少为 len，用于存放每步计算后的结果。
        //     uint64_t num_add：初始加数，每次乘法结果都会加上该值，并在循环中被更新为进位。
        //     uint64_t num_mul：乘数，in 数组的每个元素都要与它相乘。
        // 输出：
        //    返回值 uint64_t：最终的进位值（即最后一次乘加操作的高64位加上进位），用于多精度乘法的进位传递。
        // 功能说明：
        //    该函数对输入数组 in 的每个元素 in[i] 执行如下操作：
        //    1. 计算 in[i] * num_mul，得到128位结果（低64位 prod_lo，高64位 prod_hi）。
        //    2. 将 num_add 加到 prod_lo 上，结果存入 out[i]。
        //    3. 更新 num_add 为 prod_hi + (prod_lo < num_add)，即高位加上本次加法的进位。
        //    循环结束后，返回最后的 num_add，表示最高位的进位。
        inline uint64_t abs_mul_add_num64_half(const uint64_t in[], size_t len, uint64_t out[], uint64_t num_add, uint64_t num_mul)
        {
            size_t i = 0;
            uint64_t prod_lo, prod_hi;
            for (const size_t rem_len = len - len % 4; i < rem_len; i += 4)
            {
                mul64x64to128(in[i], num_mul, prod_lo, prod_hi);
                prod_lo += num_add;
                out[i] = prod_lo;
                num_add = prod_hi + (prod_lo < num_add);

                mul64x64to128(in[i + 1], num_mul, prod_lo, prod_hi);
                prod_lo += num_add;
                out[i + 1] = prod_lo;
                num_add = prod_hi + (prod_lo < num_add);

                mul64x64to128(in[i + 2], num_mul, prod_lo, prod_hi);
                prod_lo += num_add;
                out[i + 2] = prod_lo;
                num_add = prod_hi + (prod_lo < num_add);

                mul64x64to128(in[i + 3], num_mul, prod_lo, prod_hi);
                prod_lo += num_add;
                out[i + 3] = prod_lo;
                num_add = prod_hi + (prod_lo < num_add);
            }
            for (; i < len; i++)
            {
                mul64x64to128(in[i], num_mul, prod_lo, prod_hi);
                prod_lo += num_add;
                out[i] = prod_lo;
                num_add = prod_hi + (prod_lo < num_add);
            }
            return num_add;
        }

        /// @brief 2^64 base long integer multiply 64bit number, add another 64bit number to product.
        /// @param in Input long integer.
        /// @param length Number of 64-bit blocks in the input array.
        /// @param out Output long integer, equals to input * num_mul + num_add
        /// @param num_add The 64 bit number to add.
        /// @param num_mul The 64 bit number to multiply.
        /// @details
        /// The function performs multiplication and addition on a large integer represented by multiple 64-bit blocks:
        /// 1. For each block of the large integer from index 0 to `length-1`:
        ///    a. Multiply the current block `in[i]` by `num_mul`.
        ///    b. Add the current value of `num_add` to the product.
        ///    c. Store the lower 64 bits of the result in `out[i]`.
        ///    d. Update `num_add` with the higher 64 bits of the product (carry-over to the next block).
        /// 2. After processing all blocks, store the final value of `num_add` (the carry-over) in `out[length]`.
        
        //输入参数：
        //    •	const uint64_t in[]：输入的无符号 64 位整数数组，表示一个大整数，每个元素为一位（低位在前，高位在后）。
        //    •	size_t length：输入数组 in 的长度（元素个数）。
        //    •	uint64_t out[]：输出数组，长度至少为 length + 1，用于存放每步计算后的结果和最终进位。
        //    •	uint64_t num_add：初始加数，每次乘法结果都会加上该值，并在循环中被更新为进位。
        //    •	uint64_t num_mul：乘数，in 数组的每个元素都要与它相乘。
        //输出结果：
        //    •	通过 out 数组返回结果：out[i] 存储 in[i] * num_mul + num_add 的低 64 位，out[length] 存储最后一次乘加操作的高 64 位（即最终进位）。
        //    •	没有函数返回值，所有结果通过 out 数组输出。
        inline void abs_mul_add_num64(const uint64_t in[], size_t length, uint64_t out[], uint64_t num_add, uint64_t num_mul)
        {
            for (size_t i = 0; i < length; i++)
            {
                InternalUInt128 product = InternalUInt128(in[i]) * num_mul + num_add;
                out[i] = uint64_t(product);
                num_add = product.high64();
            }
            out[length] = num_add;
        }

        // in * num_mul + in_out -> in_out
        inline void mul64_sub_proc(const uint64_t in[], size_t len, uint64_t in_out[], uint64_t num_mul)
        {
            uint64_t carry = 0;
            size_t i = 0;
            for (const size_t rem_len = len - len % 4; i < rem_len; i += 4)
            {
                bool cf;
                uint64_t prod_lo, prod_hi;
                mul64x64to128(in[i], num_mul, prod_lo, prod_hi);
                prod_lo = add_half(prod_lo, in_out[i], cf);
                prod_hi += cf;
                in_out[i] = add_half(prod_lo, carry, cf);
                carry = prod_hi + cf;

                mul64x64to128(in[i + 1], num_mul, prod_lo, prod_hi);
                prod_lo = add_half(prod_lo, in_out[i + 1], cf);
                prod_hi += cf;
                in_out[i + 1] = add_half(prod_lo, carry, cf);
                carry = prod_hi + cf;

                mul64x64to128(in[i + 2], num_mul, prod_lo, prod_hi);
                prod_lo = add_half(prod_lo, in_out[i + 2], cf);
                prod_hi += cf;
                in_out[i + 2] = add_half(prod_lo, carry, cf);
                carry = prod_hi + cf;

                mul64x64to128(in[i + 3], num_mul, prod_lo, prod_hi);
                prod_lo = add_half(prod_lo, in_out[i + 3], cf);
                prod_hi += cf;
                in_out[i + 3] = add_half(prod_lo, carry, cf);
                carry = prod_hi + cf;
            }
            for (; i < len; i++)
            {
                bool cf;
                uint64_t prod_lo, prod_hi;
                mul64x64to128(in[i], num_mul, prod_lo, prod_hi);
                prod_lo = add_half(prod_lo, in_out[i], cf);
                prod_hi += cf;
                in_out[i] = add_half(prod_lo, carry, cf);
                carry = prod_hi + cf;
            }
            in_out[len] = carry;
        }

        // 小学乘法
        inline void abs_mul64_classic(const uint64_t in1[], size_t len1, const uint64_t in2[], size_t len2, uint64_t out[], uint64_t* work_begin = nullptr, uint64_t* work_end = nullptr)
        {
            const size_t out_len = get_mul_len(len1, len2);
            len1 = remove_leading_zeros(in1, len1);
            len2 = remove_leading_zeros(in2, len2);
            if (len1 < len2)
            {
                std::swap(in1, in2);
                std::swap(len1, len2); // Let in1 be the loonger one
            }
            if (0 == len2 || nullptr == in1 || nullptr == in2)
            {
                std::fill_n(out, out_len, uint64_t(0));
                return;
            }
            if (1 == len2)
            {
                abs_mul_add_num64(in1, len1, out, 0, in2[0]);
                return;
            }
            // Get enough work memory
            std::vector<uint64_t> work_mem;
            const size_t work_size = get_mul_len(len1, len2);
            if (work_begin + work_size > work_end)
            {
                work_mem.resize(work_size);
                work_begin = work_mem.data();
                work_end = work_begin + work_mem.size();
            }
            else
            {
                // Clear work_mem that may used
                std::fill_n(work_begin, work_size, uint64_t(0));
            }
            auto out_temp = work_begin;
            for (size_t i = 0; i < len1; i++)
            {
                mul64_sub_proc(in2, len2, out_temp + i, in1[i]);
            }
            std::copy(out_temp, out_temp + work_size, out);
            std::fill(out + work_size, out + out_len, uint64_t(0));
        }

        // Karatsuba 乘法
        inline void abs_mul64_karatsuba_buffered(const uint64_t in1[], size_t len1, const uint64_t in2[], size_t len2, uint64_t out[], uint64_t* buffer_begin = nullptr, uint64_t* buffer_end = nullptr)
        {
            const size_t out_len = get_mul_len(len1, len2);
            len1 = remove_leading_zeros(in1, len1);
            len2 = remove_leading_zeros(in2, len2);
            if (len1 < len2)
            {
                std::swap(in1, in2);
                std::swap(len1, len2); // Let in1 be the loonger one
            }
            if (0 == len2 || nullptr == in1 || nullptr == in2)
            {
                std::fill_n(out, out_len, uint64_t(0));
                return;
            }
            constexpr size_t KARATSUBA_THRESHOLD = 24;
            if (len2 < KARATSUBA_THRESHOLD)
            {
                abs_mul64_classic(in1, len1, in2, len2, out, buffer_begin, buffer_end);
                std::fill(out + len1 + len2, out + out_len, uint64_t(0));
                return;
            }
            // Split A * B -> (AH * BASE + AL) * (BH * BASE + BL)
            // (AH * BASE + AL) * (BH * BASE + BL) = AH * BH * BASE^2 + (AH * BL + AL * BH) * BASE + AL * BL
            // Let M = AL * BL, N = AH * BH, K1 = (AH - AL), K2 = (BH - BL), K = K1 * K2 = AH * BH - (AH * BL + AL * BH) + AL * BL
            // A * B = N * BASE^2 + (M + N - K) * BASE + M
            const size_t base_len = (len1 + 1) / 2;
            size_t len1_low = base_len, len1_high = len1 - base_len;
            size_t len2_low = base_len, len2_high = len2 - base_len;
            if (len2 <= base_len)
            {
                len2_low = len2;
                len2_high = 0;
            }
            // Get length of every part
            size_t m_len = get_mul_len(len1_low, len2_low);
            size_t n_len = get_mul_len(len1_high, len2_high);

            // Get enough buffer
            std::vector<uint64_t> buffer;
            const size_t buffer_size = m_len + n_len + get_mul_len(len1_low, len2_low);
            if (buffer_begin + buffer_size > buffer_end)
            {
                buffer.resize(buffer_size * 2);
                buffer_begin = buffer.data();
                buffer_end = buffer_begin + buffer.size();
            }
            // Set pointer of every part
            auto m = buffer_begin, n = m + m_len, k1 = n + n_len, k2 = k1 + len1_low, k = k1;

            // Compute M,N
            abs_mul64_karatsuba_buffered(in1, len1_low, in2, len2_low, m, buffer_begin + buffer_size, buffer_end);
            abs_mul64_karatsuba_buffered(in1 + base_len, len1_high, in2 + base_len, len2_high, n, buffer_begin + buffer_size, buffer_end);

            // Compute K1,K2
            len1_low = remove_leading_zeros(in1, len1_low);
            len2_low = remove_leading_zeros(in2, len2_low);
            int cmp1 = abs_difference_binary(in1, len1_low, in1 + base_len, len1_high, k1);
            int cmp2 = abs_difference_binary(in2, len2_low, in2 + base_len, len2_high, k2);
            size_t k1_len = remove_leading_zeros(k1, get_sub_len(len1_low, len1_high));
            size_t k2_len = remove_leading_zeros(k2, get_sub_len(len2_low, len2_high));

            // Compute K1*K2 = K
            abs_mul64_karatsuba_buffered(k1, k1_len, k2, k2_len, k, buffer_begin + buffer_size, buffer_end);
            size_t k_len = remove_leading_zeros(k, get_mul_len(k1_len, k2_len));

            // Combine the result
            {
                // out = M + N * BASE ^ 2 + (M + N) ^ BASE
                std::fill(out + m_len, out + base_len * 2, uint64_t(0));
                std::fill(out + base_len * 2 + n_len, out + out_len, uint64_t(0));
                std::copy(m, m + m_len, out);
                std::copy(n, n + n_len, out + base_len * 2);
                m_len = std::min(m_len, out_len - base_len);
                n_len = std::min(n_len, out_len - base_len);
                {
                    if (m_len < n_len)
                    {
                        std::swap(m_len, n_len);
                        std::swap(m, n);
                    }
                    uint8_t carry = 0;
                    size_t i = 0;
                    auto out_p = out + base_len;
                    for (; i < n_len; i++)
                    {
                        bool cf;
                        uint64_t sum = add_half(m[i], uint64_t(carry), cf);
                        carry = cf;
                        sum = add_half(n[i], sum, cf);
                        carry += cf;
                        out_p[i] = add_half(out_p[i], sum, cf);
                        carry += cf;
                    }
                    for (; i < m_len; i++)
                    {
                        bool cf;
                        uint64_t sum = add_half(m[i], uint64_t(carry), cf);
                        carry = cf;
                        out_p[i] = add_half(out_p[i], sum, cf);
                        carry += cf;
                    }
                    for (; i < out_len - base_len; i++)
                    {
                        bool cf;
                        out_p[i] = add_half(out_p[i], uint64_t(carry), cf);
                        carry = cf;
                    }
                }

                // out = M + N * BASE ^ 2 + (M + N - K) ^ BASE
                k_len = std::min(k_len, out_len - base_len);
                if (cmp1 * cmp2 > 0)
                {
                    abs_sub_binary(out + base_len, out_len - base_len, k, k_len, out + base_len);
                }
                else
                {
                    abs_add_binary_half(out + base_len, out_len - base_len, k, k_len, out + base_len);
                }
            }
        }

        inline void abs_mul64_karatsuba(const uint64_t in1[], size_t len1, const uint64_t in2[], size_t len2, uint64_t out[])
        {
            abs_mul64_karatsuba_buffered(in1, len1, in2, len2, out, nullptr, nullptr);
        }

        // NTT 平方算法
        inline void abs_sqr64_ntt(const uint64_t in[], size_t len, uint64_t out[])
        {
            using namespace HyperInt::Transform::NumberTheoreticTransform;
            if (0 == len || in == nullptr)
            {
                return;
            }
            size_t out_len = len * 2, conv_len = out_len - 1;
            size_t ntt_len = HyperInt::int_ceil2(conv_len);
            std::vector<NTT0::ModIntType> buffer1(ntt_len);
            {
                std::copy(in, in + len, buffer1.begin());
                NTT0::convolutionRecursion(buffer1.data(), buffer1.data(), buffer1.data(), ntt_len);
            };
            std::vector<NTT1::ModIntType> buffer2(ntt_len);
            {
                std::copy(in, in + len, buffer2.begin());
                NTT1::convolutionRecursion(buffer2.data(), buffer2.data(), buffer2.data(), ntt_len);
            };
            std::vector<NTT2::ModIntType> buffer3(ntt_len);
            {
                std::copy(in, in + len, buffer3.begin());
                NTT2::convolutionRecursion(buffer3.data(), buffer3.data(), buffer3.data(), ntt_len);
            };
            InternalUInt192 carry = 0;
            for (size_t i = 0; i < conv_len; i++)
            {
                carry += crt3(buffer1[i], buffer2[i], buffer3[i]);
                out[i] = uint64_t(carry);
                carry = carry.rShift64();
            }
            out[conv_len] = uint64_t(carry);
        }

        // 3NTT-CRT multiplication
        inline void abs_mul64_ntt(const uint64_t in1[], size_t len1, const uint64_t in2[], size_t len2, uint64_t out[])
        {
            if (0 == len1 || 0 == len2 || in1 == nullptr || in2 == nullptr)
            {
                return;
            }
            if (in1 == in2)
            {
                abs_sqr64_ntt(in1, len1, out); // Square
                return;
            }
            using namespace HyperInt::Transform::NumberTheoreticTransform;
            size_t out_len = len1 + len2, conv_len = out_len - 1;
            size_t ntt_len = HyperInt::int_ceil2(conv_len);
            std::vector<NTT0::ModIntType> buffer1(ntt_len);
            {
                std::vector<NTT0::ModIntType> buffer2(ntt_len);
                std::copy(in2, in2 + len2, buffer2.begin());
                std::copy(in1, in1 + len1, buffer1.begin());
                NTT0::convolutionRecursion(buffer1.data(), buffer2.data(), buffer1.data(), ntt_len);
            };
            std::vector<NTT1::ModIntType> buffer3(ntt_len);
            {
                std::vector<NTT1::ModIntType> buffer4(ntt_len);
                std::copy(in2, in2 + len2, buffer4.begin());
                std::copy(in1, in1 + len1, buffer3.begin());
                NTT1::convolutionRecursion(buffer3.data(), buffer4.data(), buffer3.data(), ntt_len);
            };
            std::vector<NTT2::ModIntType> buffer5(ntt_len);
            {
                std::vector<NTT2::ModIntType> buffer6(ntt_len);
                std::copy(in2, in2 + len2, buffer6.begin());
                std::copy(in1, in1 + len1, buffer5.begin());
                NTT2::convolutionRecursion(buffer5.data(), buffer6.data(), buffer5.data(), ntt_len);
            };
            InternalUInt192 carry = 0;
            for (size_t i = 0; i < conv_len; i++)
            {
                carry += crt3(buffer1[i], buffer3[i], buffer5[i]);
                out[i] = uint64_t(carry);
                carry = carry.rShift64();
            }
            out[conv_len] = uint64_t(carry);
        }

        // 自动根据长度选择最优算法
        inline void abs_mul64_balanced(const uint64_t in1[], size_t len1, const uint64_t in2[], size_t len2, uint64_t out[], uint64_t* work_begin = nullptr, uint64_t* work_end = nullptr)
        {
            if (len1 < len2)
            {
                std::swap(in1, in2);
                std::swap(len1, len2);
            }
            if (len2 <= 24)
            {
                abs_mul64_classic(in1, len1, in2, len2, out, work_begin, work_end);
            }
            else if (len2 <= 1536)
            {
                abs_mul64_karatsuba_buffered(in1, len1, in2, len2, out, work_begin, work_end);
            }
            else
            {
                abs_mul64_ntt(in1, len1, in2, len2, out);
            }
        }

        // 输入参数：
        //    •	const uint64_t* in1：第一个无符号 64 位整数数组（大整数的低位在前，高位在后）。
        //    •	size_t len1：in1 数组的长度（元素个数）。
        //    •	const uint64_t* in2：第二个无符号 64 位整数数组。
        //    •	size_t len2：in2 数组的长度。
        //    •	uint64_t * out：输出数组，长度至少为 len1 + len2，用于存放乘积结果（低位在前，高位在后）。
        //    •	uint64_t * work_begin（可选）：工作内存的起始指针，用于算法内部的临时空间，默认为 nullptr。
        //    •	uint64_t * work_end（可选）：工作内存的结束指针，默认为 nullptr。
        // 输出结果：
        //    •	结果通过 out 数组返回，out[0]~out[len1 + len2 - 1] 存储乘积的每一位（低位在前，高位在后）。
        //    •	函数本身无返回值（void），所有结果都写入 out。
        // 说明：
        //    •	输入的两个数组可以表示任意长度的大整数，函数会自动选择合适的乘法算法（如朴素、Karatsuba、NTT）。
        //    •	如果 work_begin 和 work_end 提供的空间不足，函数会自动分配临时内存。
        //    •	输入数组可以有前导零，函数会自动处理。
        //    •	若任一输入长度为 0，输出数组会被清零。
        inline void abs_mul64(const uint64_t in1[], size_t len1, const uint64_t in2[], size_t len2, uint64_t out[], uint64_t* work_begin = nullptr, uint64_t* work_end = nullptr)
        {
            if (len1 < len2)
            {
                std::swap(in1, in2);
                std::swap(len1, len2);
            }
            if (len2 <= 24)
            {
                abs_mul64_balanced(in1, len1, in2, len2, out, work_begin, work_end);
                return;
            }
            // Get enough work memory
            std::vector<uint64_t> work_mem;
            const size_t work_size = len2 * 3 + len1; // len1 + len2 + len2 * 2,存放结果以及平衡乘积
            if (work_begin + work_size > work_end)
            {
                work_mem.resize(work_size + len2 * 6); // 为karatsuba做准备
                work_begin = work_mem.data();
                work_end = work_begin + work_mem.size();
            }
            else
            {
                std::fill_n(work_begin, work_size, uint64_t(0));
            }
            auto balance_prod = work_begin, total_prod = balance_prod + len2 * 2;
            size_t rem = len1 % len2, i = len2;
            abs_mul64_balanced(in2, len2, in1, len2, total_prod, work_begin + work_size, work_end);
            for (; i < len1 - rem; i += len2)
            {
                abs_mul64_balanced(in2, len2, in1 + i, len2, balance_prod, work_begin + work_size, work_end);
                abs_add_binary_half(balance_prod, len2 * 2, total_prod + i, len2, total_prod + i);
            }
            if (rem > 0)
            {
                abs_mul64(in2, len2, in1 + i, rem, balance_prod, work_begin + work_size, work_end);
                abs_add_binary_half(total_prod + i, len2, balance_prod, len2 + rem, total_prod + i);
            }
            std::copy(total_prod, total_prod + len1 + len2, out);
        }

        /// @brief 大整数乘法分发器（根据尺寸选择最优算法）
        class MultiplicationSelector
        {
        public:
            using MultiplyFunc = std::function<void(const uint64_t[], size_t, const uint64_t[], size_t, uint64_t[])>;
            using ComplexityFunc = std::function<size_t(size_t, size_t)>;
            using MulFuncVector = std::vector<MultiplyFunc>;
            using CplxFuncVector = std::vector<ComplexityFunc>;
            MultiplicationSelector(const MulFuncVector& mul, const CplxFuncVector& cplx) : mul_funcs(mul), cplx_funcs(cplx) {}
            size_t selectMulFunc(size_t len1, size_t len2) const
            {
                size_t min_i = 0, min_cplx = cplx_funcs[0](len1, len2);
                for (size_t i = 1; i < cplx_funcs.size(); i++)
                {
                    auto cplx = cplx_funcs[i](len1, len2);
                    if (cplx < min_cplx)
                    {
                        min_i = i;
                        min_cplx = cplx;
                    }
                }
                return min_i;
            }
            void operator()(const uint64_t* in1, size_t len1, const uint64_t* in2, size_t len2, uint64_t* out) const
            {
                mul_funcs[selectMulFunc(len1, len2)](in1, len1, in2, len2, out);
            }

        private:
            MulFuncVector mul_funcs;
            CplxFuncVector cplx_funcs;
        };

        // 使用 Meyers Singleton 模式，通过将静态变量包装在函数内部，确保它们在第一次使用时初始化
        inline const MultiplicationSelector& getMultiplier()
        {
            static const MultiplicationSelector::MulFuncVector mul_funcs = { abs_mul64_karatsuba, abs_mul64_ntt };
            static const MultiplicationSelector::CplxFuncVector cplx_funcs = { mul_karatsuba_complexity, mul_ntt_complexity };
            static const MultiplicationSelector multiplier(mul_funcs, cplx_funcs);
            return multiplier;
        }

        template <typename NumTy, typename ProdTy>
        class DivSupporter
        {
        private:
            NumTy divisor = 0;
            NumTy inv = 0;
            int shift = 0, shift1 = 0, shift2 = 0;
            enum : int
            {
                NUM_BITS = sizeof(NumTy) * CHAR_BIT
            };

        public:
            constexpr DivSupporter(NumTy divisor_in) : divisor(divisor_in)
            {
                inv = getInv(divisor, shift);
                divisor <<= shift;
                shift1 = shift / 2;
                shift2 = shift - shift1;
            }
            // Return dividend / divisor, dividend %= divisor
            NumTy divMod(ProdTy& dividend) const
            {
                dividend <<= shift;
                NumTy r = NumTy(dividend);
                dividend = (dividend >> NUM_BITS) * inv + dividend;
                NumTy q1 = NumTy(dividend >> NUM_BITS) + 1;
                r -= q1 * divisor;
                if (r > NumTy(dividend))
                {
                    q1--;
                    r += divisor;
                }
                if (r >= divisor)
                {
                    q1++;
                    r -= divisor;
                }
                dividend = r >> shift;
                return q1;
            }

            void prodDivMod(NumTy a, NumTy b, NumTy& quot, NumTy& rem) const
            {
                ProdTy dividend = ProdTy(a << shift1) * (b << shift2);
                rem = NumTy(dividend);
                dividend = (dividend >> NUM_BITS) * inv + dividend;
                quot = NumTy(dividend >> NUM_BITS) + 1;
                rem -= quot * divisor;
                if (rem > NumTy(dividend))
                {
                    quot--;
                    rem += divisor;
                }
                if (rem >= divisor)
                {
                    quot++;
                    rem -= divisor;
                }
                rem >>= shift;
            }

            NumTy div(ProdTy dividend) const
            {
                return divMod(dividend);
            }
            NumTy mod(ProdTy dividend) const
            {
                divMod(dividend);
                return dividend;
            }

            static constexpr NumTy getInv(NumTy divisor, int& leading_zero)
            {
                constexpr NumTy MAX = all_one<NumTy>(NUM_BITS);
                leading_zero = hint_clz(divisor);
                divisor <<= leading_zero;
                ProdTy x = ProdTy(MAX - divisor) << NUM_BITS;
                return NumTy((x + MAX) / divisor);
            }
        };

        /// @brief Divides a large integer, represented as an array of 64-bit blocks, by a 64-bit divisor.
        /// @param in Input array representing the large integer to be divided. Each element of the array is a 64-bit block of the integer.
        /// @param length Number of 64-bit blocks in the input array.
        /// @param out Output array that will hold the result of the division. After division, out will represent the quotient (*this / divisor).
        /// @param divisor The 64-bit number by which the large integer is divided.
        /// @return The remainder of the division (*this % divisor), which is a 64-bit value.
        ///
        /// @details
        /// The function performs division on a large integer represented by multiple 64-bit blocks:
        /// 1. Initialize `remainder_high64bit` to 0.
        /// 2. For each block of the large integer, starting from the most significant block (index `length-1`) down to the least significant block (index 0):
        ///    a. Combine the current block `in[length]` and the current `remainder_high64bit` to form a 128-bit value.
        ///    b. Call `selfDivRem(divisor)` on this 128-bit value:
        ///       - Divide the 128-bit value by the 64-bit `divisor`.
        ///       - Update the current block in `out[llength]` with the quotient.
        ///       - Update `remainder_high64bit` with the remainder.
        /// 3. After processing all blocks, the final value of `remainder_high64bit` is the remainder of the entire division.
        /// 4. Return `remainder_high64bit`.
        
        /// @brief 短除法：将一个大整数（表示为64位块的数组）除以64位除数。
        /// @param Input数组中的表示要分割的大整数。数组的每个元素都是一个64位的整数块。
        /// @param length输入数组中64位块的数量。
        /// @param out输出数组，用于保存除法结果。除法后，out将表示商（*this/除数）。
        /// @param 除数大整数除以的64位数字。
        /// @return 除法的余数，它是一个64位的值。
        /// 
        /// @details
        /// 该函数对由多个64位块表示的大整数进行除法运算：
        /// 1.将`remainder_high64bit`初始化为0。
        /// 2.对于大整数的每个块，从最高有效块（索引'length-1'）到最低有效块（指数0）：
        ///     a.将当前块“in[length]”和当前块“remainder_high64bit”组合起来，形成一个128位的值。
        ///     b.对这个128位值调用`selfDivRem（除数）`：
        ///         -将128位值除以64位“除数”。
        ///         -用商更新'out[llength]中的当前块。
        ///         -用余数更新`remainder_high64bit`。
        /// 3.处理完所有块后，'remainder_high64bit'的最终值是整个除法的余数。
        /// 4.返回`remainder_high64bit`。
        inline uint64_t abs_div_rem_num64(const uint64_t in[], size_t length, uint64_t out[], uint64_t divisor)
        {
            uint64_t remainder_high64bit = 0;
            while (length > 0)
            {
                length--;
                InternalUInt128 n(in[length], remainder_high64bit);
                remainder_high64bit = n.selfDivRem(divisor);
                out[length] = n;
            }
            return remainder_high64bit;
        }

        // 在特定基数下，将除数规整化
        // 规整化：将除数的最高位调整到大于等于基数/2所需乘以的2的幂
        // 规整化后，除数的最高位为1，且除数的前导0都被移到最低位
        template <typename T>
        inline T divisor_normalize(T divisor[], size_t len, T base)
        {
            if (0 == len || nullptr == divisor)
            {
                return 0;
            }
            if (divisor[len - 1] >= base / 2)
            {
                return 1;
            }
            T first = divisor[len - 1];
            // “放大因子”（factor），即将最高位调整到大于等于 base / 2 所需乘以的 2 的幂。
            // 规整化
            T factor = 1;
            while (first < base / 2)
            {
                factor *= 2;
                first *= 2;
            }
            const DivSupporter<uint64_t, InternalUInt128> base_supporter(base);
            uint64_t carry = 0;
            for (size_t i = 0; i < len; i++)
            {
                InternalUInt128 temp = InternalUInt128(divisor[i]) * factor + carry;
                carry = base_supporter.divMod(temp);
                divisor[i] = T(temp);
            }
            assert(carry == 0);
            assert(divisor[len - 1] >= base / 2);
            return factor;
        }

        // 基数为 2^64 的除数规整化
        constexpr int divisor_normalize64(const uint64_t in[], size_t len, uint64_t out[])
        {
            constexpr uint64_t BASE_HALF = uint64_t(1) << 63;
            if (0 == len || nullptr == in)
            {
                return 0;
            }
            if (in[len - 1] >= BASE_HALF)
            {
                std::copy(in, in + len, out);
                return 0;
            }
            uint64_t first = in[len - 1];
            assert(first > 0);
            const int leading_zeros = hint_clz(first);
            lshift_in_word_half(in, len, out, leading_zeros);
            assert(out[len - 1] >= BASE_HALF);
            return leading_zeros;
        }

        // 只处理商的长度为dividend_len-divisor_len的情况
        // kunth除法
        inline void abs_div64_classic_core(uint64_t dividend[], size_t dividend_len, const uint64_t divisor[], size_t divisor_len, uint64_t quotient[], uint64_t* work_begin = nullptr, uint64_t* work_end = nullptr)
        {
            if (nullptr == dividend || dividend_len <= divisor_len)
            {
                return;
            }
            assert(divisor[divisor_len - 1] >= uint64_t(1) << 63); // 除数最高位为1
            assert(divisor_len > 0);
            size_t pre_dividend_len = dividend_len;
            dividend_len = remove_leading_zeros(dividend, dividend_len); // 去除前导0
            while (dividend_len < pre_dividend_len && dividend_len >= divisor_len)
            {
                size_t quot_i = dividend_len - divisor_len;
                if (abs_compare(dividend + quot_i, divisor_len, divisor, divisor_len) >= 0)
                {
                    quotient[quot_i] = 1;
                    abs_sub_binary(dividend + quot_i, divisor_len, divisor, divisor_len, dividend + quot_i);
                }
                pre_dividend_len = dividend_len;
                dividend_len = remove_leading_zeros(dividend, dividend_len); // 去除前导0
            }
            if (dividend_len < divisor_len)
            {
                return;
            }
            if (1 == divisor_len)
            {
                uint64_t rem = abs_div_rem_num64(dividend, dividend_len, quotient, divisor[0]);
                dividend[0] = rem;
                return;
            }
            // Get enough work memory
            std::vector<uint64_t> work_mem;
            const size_t work_size = divisor_len + 1;
            if (work_begin + work_size > work_end)
            {
                work_mem.resize(work_size);
                work_begin = work_mem.data();
                work_end = work_begin + work_mem.size();
            }
            const uint64_t divisor_1 = divisor[divisor_len - 1];
            const uint64_t divisor_0 = divisor[divisor_len - 2];
            const DivSupporter<uint64_t, InternalUInt128> div(divisor_1);
            size_t i = dividend_len - divisor_len;
            while (i > 0)
            {
                i--;
                uint64_t qhat = 0, rhat = 0;
                const uint64_t dividend_2 = dividend[divisor_len + i];
                const uint64_t dividend_1 = dividend[divisor_len + i - 1];
                assert(dividend_2 <= divisor_1);
                InternalUInt128 dividend_num = InternalUInt128(dividend_1, dividend_2);
                if (dividend_2 == divisor_1)
                {
                    qhat = UINT64_MAX;
                    rhat = uint64_t(dividend_num - InternalUInt128(divisor_1) * qhat);
                }
                else
                {
                    qhat = div.divMod(dividend_num);
                    rhat = uint64_t(dividend_num);
                }
                { // 3 words / 2 words refine
                    dividend_num = InternalUInt128(dividend[divisor_len + i - 2], rhat);
                    InternalUInt128 prod = InternalUInt128(divisor_0) * qhat;
                    if (prod > dividend_num)
                    {
                        qhat--;
                        prod -= divisor_0;
                        dividend_num += InternalUInt128(0, divisor_1);
                        if (dividend_num > InternalUInt128(0, divisor_1) && prod > dividend_num)
                        {
                            qhat--;
                        }
                    }
                }
                if (qhat > 0)
                {
                    auto prod = work_begin;
                    auto rem = dividend + i;
                    abs_mul_add_num64(divisor, divisor_len, prod, 0, qhat);
                    size_t len_prod = remove_leading_zeros(prod, divisor_len + 1);
                    size_t len_rem = remove_leading_zeros(rem, divisor_len + 1);
                    int count = 0;
                    while (abs_compare(prod, len_prod, rem, len_rem) > 0)
                    {
                        qhat--;
                        abs_sub_binary(prod, len_prod, divisor, divisor_len, prod);
                        len_prod = remove_leading_zeros(prod, len_prod);
                        assert(count < 2);
                        count++;
                    }
                    if (qhat > 0)
                    {
                        abs_sub_binary(rem, len_rem, prod, len_prod, rem);
                    }
                }
                quotient[i] = qhat;
            }
        }
        // 分治除法
        inline void abs_div64_recursive_core(uint64_t dividend[], size_t dividend_len, const uint64_t divisor[], size_t divisor_len, uint64_t quotient[], uint64_t* work_begin = nullptr, uint64_t* work_end = nullptr)
        {
            if (nullptr == dividend || dividend_len <= divisor_len)
            {
                return;
            }
            assert(divisor_len > 0);
            assert(divisor[divisor_len - 1] >= uint64_t(1) << 63); // 除数最高位为1
            if (divisor_len <= 32 || (dividend_len <= divisor_len + 16))
            {
                abs_div64_classic_core(dividend, dividend_len, divisor, divisor_len, quotient, work_begin, work_end);
                return;
            }
            // 被除数分段处理
            if (dividend_len > divisor_len * 2)
            {
                size_t rem = dividend_len % divisor_len, shift = dividend_len - divisor_len - rem;
                if (rem > 0)
                {
                    abs_div64_recursive_core(dividend + shift, divisor_len + rem, divisor, divisor_len, quotient + shift, work_begin, work_end);
                }
                while (shift > 0)
                {
                    shift -= divisor_len;
                    abs_div64_recursive_core(dividend + shift, divisor_len * 2, divisor, divisor_len, quotient + shift, work_begin, work_end);
                }
            }
            else if (dividend_len < divisor_len * 2)
            {
                assert(dividend_len > divisor_len);
                // 进行估商处理
                size_t shift_len = divisor_len * 2 - dividend_len; // 进行截断处理,使得截断后被除数的长度为除数的两倍,以进行估商
                size_t next_divisor_len = divisor_len - shift_len;
                size_t next_dividend_len = dividend_len - shift_len;
                size_t quot_len = next_dividend_len - next_divisor_len;
                // Get enough work memory
                std::vector<uint64_t> work_mem;
                const size_t work_size = quot_len + shift_len;
                if (work_begin + work_size > work_end)
                {
                    work_mem.resize(work_size * 3);
                    work_begin = work_mem.data();
                    work_end = work_begin + work_mem.size();
                }
                auto prod = work_begin;
                abs_div64_recursive_core(dividend + shift_len, next_dividend_len, divisor + shift_len, next_divisor_len, quotient, work_begin, work_end);
                abs_mul64(divisor, shift_len, quotient, quot_len, prod, work_begin + work_size, work_end);
                size_t prod_len = remove_leading_zeros(prod, quot_len + shift_len), rem_len = remove_leading_zeros(dividend, divisor_len);
                // 修正
                int count = 0;
                while (abs_compare(prod, prod_len, dividend, rem_len) > 0)
                {
                    abs_sub_num_binary(quotient, quot_len, uint64_t(1), quotient);
                    abs_sub_binary(prod, prod_len, divisor, shift_len, prod);
                    bool carry = abs_add_binary_half(dividend + shift_len, rem_len - shift_len, divisor + shift_len, quot_len, dividend + shift_len);
                    if (carry)
                    {
                        if (rem_len < dividend_len) // 防止溢出
                        {
                            dividend[rem_len] = 1;
                            rem_len++;
                        }
                        else
                        {
                            // 说明此时rem_len = dividend_len + 1
                            // prod_len <= quot_len + shift_len = dividend_len - divisor_len + divisor_len * 2 - dividend_len = divisor_len <= dividend_len
                            // 故此时prod_len < rem_len
                            break;
                        }
                    }
                    rem_len = remove_leading_zeros(dividend, rem_len);
                    prod_len = remove_leading_zeros(prod, prod_len);
                    assert(count < 2);
                    count++;
                }
                abs_sub_binary(dividend, rem_len, prod, prod_len, dividend);
            }
            else
            {
                // 进行两次递归处理
                size_t quot_lo_len = divisor_len / 2, quot_hi_len = divisor_len - quot_lo_len;
                size_t dividend_len1 = divisor_len + quot_hi_len, dividend_len2 = divisor_len + quot_lo_len;
                abs_div64_recursive_core(dividend + quot_lo_len, dividend_len1, divisor, divisor_len, quotient + quot_lo_len, work_begin, work_end); // 求出高位的商
                abs_div64_recursive_core(dividend, dividend_len2, divisor, divisor_len, quotient, work_begin, work_end);                             // 求出低位的商
            }
        }
        
        // 输入参数：
        //    •	uint64_t* dividend：被除数数组，表示一个大整数（低位在前，高位在后），会被修改为余数。
        //    •	size_t dividend_len：被除数数组的长度。
        //    •	const uint64_t* divisor：除数数组，表示另一个大整数（低位在前，高位在后）。
        //    •	size_t divisor_len：除数数组的长度。
        //    •	uint64_t* quotient：输出数组，存放商，长度至少为 dividend_len - divisor_len + 1。
        // 输出：
        //    •	计算结果通过 quotient（商）和 dividend（余数，低位在前，高位在后）两个数组输出。
        // 注意事项：
        //    •	计算后，quotient 存储商，dividend 被修改为余数。
        //    •	输入数组需去除前导零，且 dividend_len >= divisor_len。
        inline void abs_div64(uint64_t dividend[], size_t dividend_len, const uint64_t divisor[], size_t divisor_len, uint64_t quotient[])
        {
            std::fill_n(quotient, get_div_len(dividend_len, divisor_len), uint64_t(0));
            dividend_len = remove_leading_zeros(dividend, dividend_len);
            divisor_len = remove_leading_zeros(divisor, divisor_len);

            size_t norm_dividend_len = dividend_len + 1, norm_divisor_len = divisor_len;
            std::vector<uint64_t> dividend_v(norm_dividend_len), divisor_v(norm_divisor_len);

            auto norm_dividend = dividend_v.data(), norm_divisor = divisor_v.data();

            int leading_zero = divisor_normalize64(divisor, divisor_len, norm_divisor);
            lshift_in_word(dividend, dividend_len, norm_dividend, leading_zero);
            norm_dividend_len = remove_leading_zeros(norm_dividend, norm_dividend_len);
            // 解决商最高位为1的情况
            if (norm_dividend_len == dividend_len)
            {
                size_t quot_len = norm_dividend_len - norm_divisor_len;
                if (abs_compare(norm_dividend + quot_len, norm_divisor_len, norm_divisor, norm_divisor_len) >= 0)
                {
                    quotient[quot_len] = 1;
                    abs_sub_binary(norm_dividend + quot_len, norm_divisor_len, norm_divisor, norm_divisor_len, norm_dividend + quot_len);
                }
            }
            // 剩余的商长度为quot_len
            abs_div64_recursive_core(norm_dividend, norm_dividend_len, norm_divisor, norm_divisor_len, quotient);

            norm_divisor_len = remove_leading_zeros(norm_dividend, norm_divisor_len); // 余数在被除数上
            rshift_in_word(norm_dividend, norm_divisor_len, dividend, leading_zero);
            std::fill(dividend + norm_divisor_len, dividend + dividend_len, uint64_t(0));
        }

    } // namespace Arithmetic
} // namespace HyperInt
#endif



class BigInt {
private:
    std::vector<uint64_t> data;  // 存储大整数的数据（小端序：data[0]是最低位）
    bool is_negative;            // 符号标志：true表示负数

    // 移除前导零
    void remove_leading_zeros() {
        while (data.size() > 1 && data.back() == 0) {
            data.pop_back();
        }
        if (data.size() == 1 && data[0] == 0) {
            is_negative = false;  // 0没有负数
        }
    }

    // 无符号比较（不考虑符号）
    int unsigned_compare(const BigInt& other) const {
        if (data.size() != other.data.size()) {
            return data.size() < other.data.size() ? -1 : 1;
        }
        for (int i = static_cast<int>(data.size()) - 1; i >= 0; --i) {
            if (data[i] != other.data[i]) {
                return data[i] < other.data[i] ? -1 : 1;
            }
        }
        return 0;
    }

    // 无符号加法
    static BigInt unsigned_add(const BigInt& a, const BigInt& b) {
        size_t max_len = std::max(a.data.size(), b.data.size());
        size_t result_len = max_len + 1;
        std::vector<uint64_t> result(result_len, 0);

        HyperInt::Arithmetic::abs_add_binary(
            a.data.data(), a.data.size(),
            b.data.data(), b.data.size(),
            result.data()
        );

        BigInt res;
        res.data = result;
        res.remove_leading_zeros();
        return res;
    }

    // 无符号减法（要求a >= b）
    static BigInt unsigned_subtract(const BigInt& a, const BigInt& b) {
        if (a.unsigned_compare(b) < 0) {
            throw std::invalid_argument("a must be greater than or equal to b");
        }

        size_t max_len = std::max(a.data.size(), b.data.size());
        std::vector<uint64_t> result(max_len, 0);

        HyperInt::Arithmetic::abs_sub_binary(
            a.data.data(), a.data.size(),
            b.data.data(), b.data.size(),
            result.data()
        );

        BigInt res;
        res.data = result;
        res.remove_leading_zeros();
        return res;
    }

    // 无符号乘法
    static BigInt unsigned_multiply(const BigInt& a, const BigInt& b) {
        if (a.is_zero() || b.is_zero()) {
            return BigInt(0);
        }

        size_t result_len = a.data.size() + b.data.size();
        std::vector<uint64_t> result(result_len, 0);

        HyperInt::Arithmetic::abs_mul64(
            a.data.data(), a.data.size(),
            b.data.data(), b.data.size(),
            result.data()
        );

        BigInt res;
        res.data = result;
        res.remove_leading_zeros();
        return res;
    }

    // 无符号除法（返回商和余数）
    static std::pair<BigInt, BigInt> unsigned_divide(const BigInt& a, const BigInt& b) {
        if (b.is_zero()) {
            throw std::invalid_argument("Division by zero");
        }

        if (a.unsigned_compare(b) < 0) {
            return { BigInt(0), a };
        }

        std::vector<uint64_t> dividend = a.data;
        std::vector<uint64_t> divisor = b.data;
        std::vector<uint64_t> quotient(dividend.size() - divisor.size() + 1, 0);
        std::vector<uint64_t> remainder(dividend.size(), 0);

        HyperInt::Arithmetic::abs_div64(
            dividend.data(), dividend.size(),
            divisor.data(), divisor.size(),
            quotient.data()
        );

        // 余数存储在dividend中
        remainder = dividend;

        BigInt quot;
        quot.data = quotient;
        quot.remove_leading_zeros();

        BigInt rem;
        rem.data = remainder;
        rem.remove_leading_zeros();

        return { quot, rem };
    }

public:
    // 构造函数
    BigInt() : data(1, 0), is_negative(false) {}

    BigInt(int64_t value) {
        if (value == 0) {
            data = { 0 };
            is_negative = false;
        }
        else {
            is_negative = value < 0;
            uint64_t abs_value = static_cast<uint64_t>(std::abs(value));
            data = { abs_value };
        }
    }

    BigInt(const std::string& str) {
        if (str.empty()) {
            data = { 0 };
            is_negative = false;
            return;
        }

        // 处理符号
        size_t start = 0;
        is_negative = false;
        if (str[0] == '-') {
            is_negative = true;
            start = 1;
        }
        else if (str[0] == '+') {
            start = 1;
        }

        // 移除前导零
        while (start < str.size() - 1 && str[start] == '0') {
            start++;
        }

        // 处理空字符串或仅符号的情况
        if (start >= str.size()) {
            data = { 0 };
            is_negative = false;
            return;
        }

        // 从字符串解析数字
        BigInt result(0);
        BigInt base(10);

        for (size_t i = start; i < str.size(); ++i) {
            if (!std::isdigit(str[i])) {
                throw std::invalid_argument("Invalid character in number string");
            }

            int digit = str[i] - '0';
            result = result * base + BigInt(digit);
        }

        data = result.data;
        remove_leading_zeros();
    }

    // 拷贝构造函数
    BigInt(const BigInt& other) = default;

    // 移动构造函数
    BigInt(BigInt&& other) noexcept = default;

    // 赋值运算符
    BigInt& operator=(const BigInt& other) = default;

    // 判断是否为零
    bool is_zero() const {
        return data.size() == 1 && data[0] == 0;
    }

    // 转换为字符串
    [[nodiscard]] std::string to_string() const {
        if (is_zero()) {
            return "0";
        }

        // 10^19进制基数 (1e19 < 2^64)
        constexpr uint64_t base = 10000000000000000000ULL;
        constexpr int base_digits = 19;

        BigInt num = *this;
        num.is_negative = false;  // 使用绝对值

        // 存储各段的余数（从低位到高位）
        std::vector<uint64_t> chunks;

        // 高效除法：每次处理19位数字
        while (!num.is_zero()) {
            std::vector<uint64_t> quotient(num.data.size());
            uint64_t remainder = HyperInt::Arithmetic::abs_div_rem_num64(
                num.data.data(), num.data.size(),
                quotient.data(), base
            );

            chunks.push_back(remainder);
            num.data = quotient;
            num.remove_leading_zeros();
        }

        // 处理符号和第一段（最高位）
        std::string result;
        if (is_negative) {
            result += '-';
        }

        // 最后一段不需要前导零
        result += std::to_string(chunks.back());

        // 处理剩余段（每段固定19位）
        for (auto it = chunks.rbegin() + 1; it != chunks.rend(); ++it) {
            char buf[base_digits + 1];
            snprintf(buf, sizeof(buf), "%019" PRIu64, *it);
            result += buf;
        }

        return result;
    }

    // 一元负号运算符
    BigInt operator-() const {
        BigInt result = *this;
        if (!is_zero()) {
            result.is_negative = !is_negative;
        }
        return result;
    }

    // 加法运算符
    BigInt operator+(const BigInt& other) const {
        // 处理符号不同的情况
        if (is_negative != other.is_negative) {
            if (is_negative) {
                return other - (-*this);
            }
            else {
                return *this - (-other);
            }
        }

        // 同号情况
        BigInt result = unsigned_add(*this, other);
        result.is_negative = is_negative;
        return result;
    }

    // 减法运算符
    BigInt operator-(const BigInt& other) const {
        // 处理符号不同的情况
        if (is_negative != other.is_negative) {
            return *this + (-other);
        }

        // 同号情况
        int cmp = unsigned_compare(other);
        if (cmp == 0) {
            return BigInt(0);
        }

        if (cmp > 0) {
            BigInt result = unsigned_subtract(*this, other);
            result.is_negative = is_negative;
            return result;
        }
        else {
            BigInt result = unsigned_subtract(other, *this);
            result.is_negative = !is_negative;
            return result;
        }
    }

    // 乘法运算符
    BigInt operator*(const BigInt& other) const {
        if (is_zero() || other.is_zero()) {
            return BigInt(0);
        }

        BigInt result = unsigned_multiply(*this, other);
        result.is_negative = is_negative != other.is_negative;
        return result;
    }

    // 除法运算符（整数除法）
    BigInt operator/(const BigInt& other) const {
        std::pair<BigInt, BigInt> div_result = unsigned_divide(*this, other);
        BigInt quotient = div_result.first;
        BigInt remainder = div_result.second;
        quotient.is_negative = is_negative != other.is_negative;
        return quotient;
    }

    // 取模运算符
    BigInt operator%(const BigInt& other) const {
        std::pair<BigInt, BigInt> div_result = unsigned_divide(*this, other);
        BigInt quotient = div_result.first;
        BigInt remainder = div_result.second;
        remainder.is_negative = is_negative;
        return remainder;
    }

    // 幂运算
    BigInt pow(uint64_t exponent) const {
        if (exponent == 0) {
            return BigInt(1);
        }

        if (is_zero()) {
            return BigInt(0);
        }

        BigInt base = *this;
        BigInt result(1);

        while (exponent > 0) {
            if (exponent & 1) {
                result = result * base;
            }
            base = base * base;
            exponent /= 2;
        }

        return result;
    }

    // 比较运算符
    bool operator==(const BigInt& other) const {
        return is_negative == other.is_negative && data == other.data;
    }

    bool operator!=(const BigInt& other) const {
        return !(*this == other);
    }

    bool operator<(const BigInt& other) const {
        if (is_negative != other.is_negative) {
            return is_negative;
        }

        int cmp = unsigned_compare(other);
        if (cmp == 0) return false;

        return is_negative ? (cmp > 0) : (cmp < 0);
    }

    bool operator<=(const BigInt& other) const {
        return *this < other || *this == other;
    }

    bool operator>(const BigInt& other) const {
        return !(*this <= other);
    }

    bool operator>=(const BigInt& other) const {
        return !(*this < other);
    }

    // 输出运算符
    friend std::ostream& operator<<(std::ostream& os, const BigInt& num) {
        os << num.to_string();
        return os;
    }

    // 转换为int（如果可能）
    [[nodiscard]] int to_int() const {
        if (is_zero()) return 0;

        if (data.size() > 10) {// 太大了
            return is_negative ? INT_MIN : INT_MAX;
        }

        return (int)data[0] * (is_negative ? -1 : 1);
    }

};


