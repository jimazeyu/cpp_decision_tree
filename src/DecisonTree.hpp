#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include"DataFrame.hpp"
#include"queue"
#include"algorithm"
#include"math.h"
using namespace std;

//分隔符
void log()
{
    cout<<"--------------------------------------------------------"<<endl;
}
struct TreeNode
{
    TreeNode *left=NULL;
    TreeNode *right=NULL;
    double threshold;//决策树划分阈值
    string feature;//用于划分的标签
    vector<single_data> DataSet;//single_data是定义在DataFrame中的数据集合类，代表一条记录,DataSet是节点包含的所有数据
    double entropy;//记录信息熵
    map<int,int> propobility;//每种类别的可能性
};
class DecisionTree{
    public:
        DecisionTree(vector<single_data> DS,vector<string> fetures);//构造函数
        vector<string> features;//所有的特征
        TreeNode* root;//决策树根节点
        //计算某个特征下，要获得最大的信息增益，选择的阈值以及信息增益大小
        friend double CalculateGain(vector<single_data> DataSet,string feature,double entropy,double &threshold,double &LeftEntropy,double &RightEntropy);
        //训练决策树
        void BuildTree(TreeNode *fa,int depth,int MaxDepth,int MaxNums);
        //预测,返回每种可能性
        friend map<int,int> predict(TreeNode *fa,single_data data);
        //后剪枝
        friend void PostPruning(TreeNode *fa,vector<single_data> DataSet);
};

//构造函数
DecisionTree::DecisionTree(vector<single_data> DataSet,vector<string> fetures)
{
    root=new TreeNode;
    root->DataSet=DataSet;
    features=fetures;
    //初始化熵
    map<int,int> LabelMap;//统计不同labels包含的个数
    int total=DataSet.size();//节点包含数据总数
    for(auto data:DataSet)
    {
        LabelMap[data.label]++;
    }
    double TmpEntropy=0;
    for(auto lb:LabelMap)
    {
        double p=1.0*lb.second/total;
        TmpEntropy+=-p*log2(p);
    }
    root->entropy=TmpEntropy;
    cout<<"loading data "<<total<<" records"<<endl;
    cout<<"initial entropy:"<<TmpEntropy<<endl;
    cout<<endl;
}

//计算确定特征下的最大信息增益
double CalculateGain(vector<single_data> DataSet,string feature,double entropy,double &threshold,double &LeftEntropy,double &RightEntropy)
{
    map<int,int> LabelMap;//统计不同labels包含的个数
    int total=DataSet.size();//节点包含数据总数
    for(auto data:DataSet)
    {
        LabelMap[data.label]++;
    }
    // //函数内定义的比较函数,利用C++11新特性
    auto comp = [&](single_data A,single_data B){return A.feature_to_value[feature]<B.feature_to_value[feature];};
    sort(DataSet.begin(),DataSet.end(),comp);//通过feature进行排序
    map<int,int> TmpLabelMap;//随着阈值更新，实时记录labels个数，扫一遍计算出信息熵
    double MinEntropy=1e6;
    int LeftNums=0;
    int RightNums=0;
    for(int i=1;i<DataSet.size();i++)//因为是排序后的，循环的其实是threshold，理论上应该二分找不同的值，偷懒了
    {
        single_data data=DataSet[i];
        TmpLabelMap[data.label]++;
        if(abs(data.feature_to_value[feature]-DataSet[i-1].feature_to_value[feature])<1e-6)continue;     
        double TmpLeftEntropy=0;
        double TmpRightEntropy=0;
        for(auto lb:TmpLabelMap)//因为是二分决策树，一个节点分为两部分，更新阈值左边每种标签的个数
        {
            double left=lb.second;
            double right=LabelMap[lb.first]-lb.second;
            double p_left=1.0*left/i; //分到左侧的该label下的数据占左侧比例 
            double p_right=1.0*right/(total-i);//分到右侧的该label下的数据占右侧比例 
            TmpLeftEntropy=-p_left*(log2(p_left));
            TmpRightEntropy=-p_right*(log2(p_right));
          
        }
        if(feature=="Age"&&total<100)
        {
            //cout<<1;
        }
        double weight1=1.0*i/total;//左子树权重
        double weight2=1.0*(total-i)/total;//右子树权重
        if(weight1*TmpLeftEntropy+weight2*TmpRightEntropy<MinEntropy)
        {
            LeftNums=i;
            RightNums=total-i;
            MinEntropy=weight1*TmpLeftEntropy+weight2*TmpRightEntropy;
            RightEntropy=TmpRightEntropy;
            LeftEntropy=TmpLeftEntropy;
            threshold=data.feature_to_value[feature]-0.000001;//防止精度损失
        }  
    }
    double gain=entropy-MinEntropy;
    return gain;
}

//训练决策树
void DecisionTree::BuildTree(TreeNode *fa,int depth,int MaxDepth,int MaxNums)
{
    if(depth>=MaxDepth)return;
    vector<single_data> DS=fa->DataSet;
    double threshold=0;
    double gain=0;
    double LeftEntropy;
    double RightEntropy;
    string DivideFeature;
    for(string feature:features)
    {
        double TmpThreshold;
        double TmpLeftEntropy;
        double TmpRightEntropy;
        double TmpGain=CalculateGain(DS,feature,fa->entropy,TmpThreshold,TmpLeftEntropy,TmpRightEntropy);
        if(TmpGain>gain)
        {
            gain=TmpGain;
            LeftEntropy=TmpLeftEntropy;
            RightEntropy=TmpRightEntropy;
            threshold=TmpThreshold;
            DivideFeature=feature;
        }
    }
    TreeNode *LeftSon=new TreeNode;
    TreeNode *RightSon=new TreeNode;
    LeftSon->entropy=LeftEntropy;
    RightSon->entropy=RightEntropy;
    map<int,int> propobility;//记录每个标签样本个数，除总数即为概率
    for(auto data:DS)
    {
        if(data.feature_to_value[DivideFeature]<threshold)
        {
            LeftSon->DataSet.push_back(data);
            LeftSon->propobility[data.label]++;
        }
        else
        {
            RightSon->DataSet.push_back(data);
            RightSon->propobility[data.label]++;
        }
    }

    if(LeftSon->DataSet.size()>=MaxNums&&RightSon->DataSet.size()>=MaxNums)
    {
        log();
        cout<<"depth:"<<depth<<endl;
        cout<<"feature:"<<DivideFeature<<endl;
        cout<<"threshold:"<<threshold<<endl;
        cout<<"leftson:"<<LeftSon->DataSet.size()<<endl;
        cout<<"rightson:"<<RightSon->DataSet.size()<<endl;
        cout<<"gain:"<<gain<<endl;
        log();
        fa->threshold=threshold;
        fa->left=LeftSon;
        fa->feature=DivideFeature;
        BuildTree(LeftSon,depth+1,MaxDepth,MaxNums);
        fa->right=RightSon;
        fa->feature=DivideFeature;
        BuildTree(RightSon,depth+1,MaxDepth,MaxNums);        
    }
}

//预测
map<int,int> predict(TreeNode *fa,single_data data)
{
    if(fa->left)
    {
        if(data.feature_to_value[fa->feature]<fa->threshold)
        {
            return predict(fa->left,data);
        }
        else 
        {
            return predict(fa->right,data);
        }
    }
    else 
    {
        return fa->propobility;
    }
}

//后剪枝
void PostPruning(TreeNode *fa,vector<single_data> DataSet)
{
    if(!fa->left)return;//已经是叶子节点了
    int LeftSons,RightSons;
    double LeftEntropy=0,RightEntropy=0;
    vector<single_data> DataSet1;
    vector<single_data> DataSet2;
    for(auto data:DataSet)
    {
        if(data.feature_to_value[fa->feature]<fa->threshold)DataSet1.push_back(data);
        else DataSet2.push_back(data);
    }
    //左节点信息熵
    {
        map<int,int> LabelMap;//统计不同labels包含的个数
        int total=DataSet1.size();//节点包含数据总数
        for(auto data:DataSet1)
        {
            LabelMap[data.label]++;
        }
        double TmpEntropy=0;
        for(auto lb:LabelMap)
        {
            double p=1.0*lb.second/total;
            TmpEntropy+=-p*log2(p);
        }
        LeftSons=total;
        LeftEntropy=TmpEntropy;
    }
    //右节点信息熵
    {
        map<int,int> LabelMap;//统计不同labels包含的个数
        int total=DataSet2.size();//节点包含数据总数
        for(auto data:DataSet2)
        {
            LabelMap[data.label]++;
        }
        double TmpEntropy=0;
        for(auto lb:LabelMap)
        {
            double p=1.0*lb.second/total;
            TmpEntropy+=-p*log2(p);
        }
        RightSons=total;
        RightEntropy=TmpEntropy;
    }
    int total=RightSons+LeftSons;
    double weight1=1.0*RightSons/total;
    double weight2=1.0*LeftSons/total;
    if(fa->entropy-weight1*LeftEntropy+weight2*RightEntropy<0)
    {
        cout<<"cut:"<<fa->entropy-weight1*LeftEntropy+weight2*RightEntropy<<endl;
        //没释放内存
        fa->left=NULL;
        fa->right=NULL;
        return;
    }
    //cout<<fa->entropy<<" "<<weight1*LeftEntropy<<" "<<weight2*RightEntropy<<" "<<fa->entropy-weight1*LeftEntropy+weight2*RightEntropy<<endl;
    PostPruning(fa->left,DataSet1);
    PostPruning(fa->right,DataSet2);
}

#endif