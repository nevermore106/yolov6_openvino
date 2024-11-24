#include "header.h"
#include <random>
using namespace std;

void Deal::Get_process(vector<Detection>&output){
  map.clear();
  int n = output.size();
  int cnt = 1;
  if(!n)
    return;
  for(int i = 1;i<n;i++){
    if(!Is_close(output[i-1],output[i])||!Is_Same_Line(output[i-1],output[i]))
      break;
    cnt++;
  }
  for(int i = 0;i<cnt;i++){
    for(int j = i+1;j<cnt;j++){
      if(output[i].box.x>output[j].box.x)
        swap(output[i],output[j]);
    }
  }
  for(int i = 0;i<cnt;i++){
    map.push_back(output[i].class_id+'0');
  }
  for(int i = 0;i<cnt;i++)
    cout<<map[i];
  cout<<endl;
}


int Deal::Euclidean_distance(vector<int>a,vector<int>b){
  if(a.size()!=2||b.size()!=2)
    return -1;
  return sqrt(pow(a[0]-b[0],2)+pow(a[1]-b[1],2));
}

bool Deal::Is_close(Detection &pre,Detection &cur){
  if(Euclidean_distance({pre.box.x,pre.box.y},{cur.box.x,cur.box.y})>pre.box.width*1.5f)
    return false;
  return true;
}

bool Deal::Is_Same_Line(Detection &pre,Detection &cur){
  int threshold = 0;
  mode_==0?threshold = 10:threshold = 20;
  if(abs(pre.box.y-cur.box.y)>10)
    return false;
  return true;
}

std::string Deal::Wrong_Number_Filter(vector<vector<char>>mat){
  int mx = 0;
  // for(int i =0;i<mat.size();i++){
  //   int n = mat[i].size();
  //   mx = max(mx,n);
  // }
  // for(int i =0;i<mat.size();i++){
  //   if(mat[i].size() == mx)
  //     return mat[i];
  // }
  // return {};
  for (int i = 0;i<mat.size();i++) {
    int temp = 0;
    for(int j = 0;j < mat[i].size();j++) {
      temp *= 10;
      temp += mat[i][j] - '0';
    }
    mx = max(mx,temp);
  }
  return to_string(mx);
}

void Number_sort(vector<int>&map){

}

