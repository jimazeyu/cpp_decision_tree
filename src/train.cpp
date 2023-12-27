#include"DecisonTree.hpp"
#include"DataFrame.hpp"
using namespace std;
//计算正负样本个数
void CalculateAccuracy(vector<single_data> dataset,DecisionTree tree)
{
    int correct=0;
    int wrong=0;
    for(auto data:dataset)
    {
        map<int,int> res=predict(tree.root,data);
        double pb=0;
        double ans;
        for(auto prob:res)
        {
            if(prob.second>pb)
            {
                pb=prob.second;
                ans=prob.first;
            }
        }
        if(ans==data.label)correct++;
        else
        {
            wrong++;
        } 
        //cout<<ans<<" "<<data.label<<endl;
    }
    cout<<"correct:"<<correct<<" wrong:"<<wrong<<endl;
}
int main()
{
    DataFrame df_train=read_csv("train.csv");//读入训练数据
    df_train.display_datas();//打印数据
    string label="Survived";//label标签
    vector<string> features=df_train.get_all_numerical_features(label);
    vector<single_data> dataset=df_train.numerical_dataset(label);
    DecisionTree tree(dataset,features);//初始化决策树
    tree.BuildTree(tree.root,1,5,1);//开始训练，决策树最大深度为5，最小节点大小为1；

    //读入测试集合
    DataFrame df_test=read_csv("test.csv");
    vector<single_data> test_dataset=df_test.numerical_dataset(label);  //后剪枝      

    CalculateAccuracy(test_dataset,tree);
    //读入测试数据用于后剪枝，提高泛化能力
    DataFrame df_cut=read_csv("cut.csv");
    vector<single_data> dataset2=df_cut.numerical_dataset(label);  //后剪枝  
    PostPruning(tree.root,dataset2);

    CalculateAccuracy(test_dataset,tree);
    return 0;
}