#pragma once
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/signal.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <string>
 
const long long RecvBufferLen = 1024;   //设置接收数据缓冲区大小
 
typedef enum
{
    _2400,
    _4800,
    _9600,
    _19200,
    _38400,
    _57600,
    _115200,
    _460800,
}E_BaudRate;  //波特率
 
typedef enum
{
    _5,
    _6,
    _7,
    _8,
}E_DataSize;  //数据位
 
typedef enum
{
    None,
    Odd,
    Even,
}E_Parity;  //校验位
 
typedef enum
{
    _1,
    _2,
}E_StopBit;  //停止位
 
 
class Serial
{
public:
    Serial();
    ~Serial();
 
    int OpenSerial(std::string SerialID, E_BaudRate Bps, E_DataSize DataSize, E_Parity Parity, E_StopBit StopBit);
 
    int Send(unsigned char *Buff, int length,int id);
    int Recv(unsigned char *Buff, int length);
 
    int Close();
    int nSerialID;  //串口
    unsigned char RecvBuf_[4096];
 
private:
    void RunConnect();
    void RunRecv();
    int RefreshBuffer(unsigned char *pBuf, int Len, bool RecvTypet);
 
private:
 
    bool b_OpenSign;   //串口打开标志
 
    struct termios ProtoOpt;   //存放串口原始配置
};
