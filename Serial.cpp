#include "Serial.h"
#include <thread>
#include <iostream>
 
Serial::Serial()
{
    b_OpenSign = false;
    nSerialID = 0;
}
 
Serial::~Serial()
{
    Close();
}
 
 
int Serial::OpenSerial(std::string SerialID, E_BaudRate Bps, E_DataSize DataSize, E_Parity Parity, E_StopBit StopBit)
{
    // Close();
    nSerialID = open( SerialID.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (-1 == nSerialID)
    {
        /* 不能打开串口一*/
        std::string str = SerialID + " open fail !!!";
        perror(str.c_str());
        return -1;
    }
    struct termios Opt;
    tcgetattr(nSerialID, &ProtoOpt);  //获取设备当前的设置
 
    Opt = ProtoOpt;//colorImage
 
    /*设置输入输出波特率*/
    switch(Bps)
    {
        case E_BaudRate::_2400:
            cfsetispeed(&Opt,B2400);
            cfsetospeed(&Opt,B2400);
            break;
        case E_BaudRate::_4800:
            cfsetispeed(&Opt,B4800);
            cfsetospeed(&Opt,B4800);
            break;
        case E_BaudRate::_9600:
            cfsetispeed(&Opt,B9600);
            cfsetospeed(&Opt,B9600);
            break;
        case E_BaudRate::_19200:
            cfsetispeed(&Opt,B19200);
            cfsetospeed(&Opt,B19200);
            break;
        case E_BaudRate::_38400:
            cfsetispeed(&Opt,B38400);
            cfsetospeed(&Opt,B38400);
            break;
        case E_BaudRate::_57600:
            cfsetispeed(&Opt,B57600);
            cfsetospeed(&Opt,B57600);
            break;
        case E_BaudRate::_115200:
            cfsetispeed(&Opt,B115200);
            cfsetospeed(&Opt,B115200);
            break;
        case E_BaudRate::_460800:
            cfsetispeed(&Opt,B460800);
            cfsetospeed(&Opt,B460800);
            break;
        default		:
            printf("Don't exist baudrate %d !\n",Bps);
            return (-1);
    }
 
    /*设置数据位*/
    Opt.c_cflag &= (~CSIZE);
    switch( DataSize )
    {
        case E_DataSize::_5:
            Opt.c_cflag |= CS5;
            break;
        case E_DataSize::_6:
            Opt.c_cflag |= CS6;
        case E_DataSize::_7:
            Opt.c_cflag |= CS7;
            break;
        case E_DataSize::_8:
            Opt.c_cflag |= CS8;
            break;
        default:
            /*perror("Don't exist iDataSize !");*/
            printf("Don't exist DataSize %d !\n",DataSize);
            return (-1);
    }
 
    /*设置校验位*/
    switch( Parity )
    {
        case E_Parity::None:					/*无校验*/
            Opt.c_cflag &= (~PARENB);
            break;
        case E_Parity::Odd:					/*奇校验*/
            Opt.c_cflag |= PARENB;
            Opt.c_cflag |= PARODD;
            Opt.c_iflag |= (INPCK | ISTRIP);
            break;
        case E_Parity::Even:					/*偶校验*/
            Opt.c_cflag |= PARENB;
            Opt.c_cflag &= (~PARODD);
            Opt.c_iflag |= (INPCK | ISTRIP);
            break;
        default:
            /*perror("Don't exist cParity  !");*/
            printf("Don't exist Parity %c !\n",Parity);
            return (-1);
    }
 
    /*设置停止位*/
 
    switch( StopBit )
    {
        case E_StopBit::_1:
            Opt.c_cflag &= (~CSTOPB);
            break;
        case E_StopBit::_2:
            Opt.c_cflag |= CSTOPB;
            break;
        default:
            printf("Don't exist iStopBit %d !\n",StopBit);
            return (-1);
    }
 
    //如果只是串口传输数据，而不需要串口来处理，那么使用原始模式(Raw Mode)方式来通讯，设置方式如下：
    Opt.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    Opt.c_oflag &= ~OPOST;
 
    tcflush(nSerialID,TCIOFLUSH);		/*刷新输入队列(TCIOFLUSH为刷新输入输出队列)*/
 
    Opt.c_cc[VTIME] = 0;	/*设置等待时间*/
    Opt.c_cc[VMIN] = 0;	/*设置最小字符*/
 
    int Result = tcsetattr(nSerialID,TCSANOW,&Opt);  //使这些设置生效
 
    if( Result )
    {
        perror("Set new terminal description error !");
        return (-1);
    }
 
    b_OpenSign = true;
 
    // RunRecv();
 
    return 0;
}
 
int Serial::Send(unsigned char *Buff, int length, int id)
{
    int iLen = 0;
    if(length <= 0)
    {
        printf("Send byte number error !\n");
        return -1;
    }
 
    iLen = write(id,Buff,length);
 
    return iLen;
}
 
int Serial::Recv(unsigned char *Buff, int length)
{
    int res =  RefreshBuffer(Buff, length, true);
 
    return res;
}
 
int Serial::Close()
{
    if(nSerialID > 0)
    {
        tcsetattr (nSerialID, TCSADRAIN, &ProtoOpt);  //恢复原始串口配置
    }
    close(nSerialID);
    b_OpenSign = false;
}
 
 
void Serial::RunRecv()
{
    std::thread ThRecv	= std::thread
            {
                    [&]()
                    {
                        // while (b_OpenSign)
                        // {
                        //     usleep(10*1000);
                        //     if((nSerialID < 0))
                        //     {
                        //         continue;
                        //     }
 
                        //     memset(RecvBuf_, 0, 4096);
 
                        //     int res = read(nSerialID, RecvBuf_, sizeof(RecvBuf_));
                        //     if(res > 0)
                        //     {
                        //         RefreshBuffer(RecvBuf_, res, false);
                        //     }
                        // }
                    }
            };
 
    ThRecv.detach();
}
 
 
int Serial::RefreshBuffer(unsigned char *pBuf, int Len, bool RecvTypet)
{
    static unsigned char  Buffer[RecvBufferLen + 1] = {0};
    static int 					nSum=0;				    //	缓冲区中数据总长度
    signed int 					nStop=0;
 
    int ren = 0;
 
    if(false == RecvTypet)
    {
        //************************ 将接收到的数据加入缓冲区中 ************************/
        //std::cout<<"recv = "<< Len <<std::endl;
 
        if((Len + nSum) <= RecvBufferLen)		//	总长度小于1K
        {
            memcpy(&Buffer[nSum], pBuf, Len);
            nSum = Len + nSum;
        }
        else
        {
            if(Len <= RecvBufferLen)			//	拷贝满1K空间，丢弃掉aucT[0]开始的字符，并进行填充，!!!!!!!!!!!
            {
                memcpy(Buffer, pBuf, Len);
                nSum = Len;
            }
            else						//	本次接收到的数据长度大于1K
            {
                memcpy(Buffer, pBuf + (Len - RecvBufferLen), RecvBufferLen);
                nSum = RecvBufferLen;
            }
        }
        //std::cout<<"----> nSum = "<< nSum <<std::endl;
        ren = 0;
    }
    else
    {
        if(Len <= 0)
        {
            return -1;
        }
        if(nSum <= 0)
        {
            return 0;
        }
 
        if(Len <= nSum)
        {
            memcpy(pBuf, Buffer, Len);
 
            nStop =  Len;
            ren = Len;
        }
        else
        {
            memcpy(pBuf, Buffer, nSum);
 
            nStop = nSum;
            ren = nSum;
        }
 
 
        //************ 移动取出数据 ***************/
        if(nStop==0)
        {
            return 0;
        }
        else if(nSum > nStop)	  // 把没有解析到的数据移动到最开始位置
        {
            for(int i=0; i<(nSum-nStop); i++)
            {
                Buffer[i] = Buffer[nStop + i];
            }
            nSum = nSum - nStop;
        }
        else if(nSum == nStop)
        {
            nSum = 0;
        }
 
    }
 
    return ren;
}
