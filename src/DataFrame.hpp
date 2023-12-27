/*
 * @Descripttion: 
   1.定义了一个series类，包含数据标签以及数据
   2.定义一个DataFrame类，包含多条Series，并提供相关数据处理函数
   3.提供从csv文件中读取数据，并自动推断数据类型，将每列生成一个Series，并组成DataFrame，
   4.将数值类型的特征组合成矩阵数组输出，供后续模型使用
   5.one-hot编码，将字符串类型Series转为n(不同值个数)个01Series
 * @version: 
 * @Author: jimazeyu
 * @Date: 1221-10-23 14:16:11
 * @LastEditors: jimazeyu
 * @LastEditTime: 2021-11-21 00:28:52
 */

#ifndef DATAFRAME_H
#define DATAFRAME_H
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <typeinfo>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string.h>
#include <set>
using namespace std;
//Series
template <typename T>
class Series
{
private:
   string feature_name; //name of feature
   vector<T> data;
   friend class DataFrame;

public:
   Series() = default;
   Series(string name) : feature_name(name){};
   Series(string name, vector<T> array) : feature_name(name), data(array){};
   ~Series(){};
   T &operator[](const int &);
   void append(T);
   int size() { return data.size(); }
   void set_feature_name(string name) { feature_name = name; }
   string get_feature_name() { return feature_name; }
   vector<T> get_data() { return data; }
};
template <typename T>
T &Series<T>::operator[](const int &id)
{
   return data[id];
}
template <typename T>
void Series<T>::append(T dt)
{
   data.push_back(dt);
}

//variable data for appending datas
struct meta_element
{
   vector<string> int_feature_names;
   vector<int> int_datas;
   vector<string> double_feature_names;
   vector<double> double_datas;
   vector<string> string_feature_names;
   vector<string> string_datas;
};

struct single_data //single record of data
{
   map<string, double> feature_to_value;
   int label;
};

//DataFrame
class DataFrame
{
private:
   int totol_data_numbers;
   vector<string> feature_names;
   map<string, pair<string, int>> name_to_pos; //using feature names to find series
   vector<Series<int>> int_arrays;
   vector<Series<double>> double_arrays;
   vector<Series<string>> string_arrays;

public:
   DataFrame() { totol_data_numbers = 0; }
   ~DataFrame(){};
   //add cols
   void add_feature(Series<int>);
   void add_feature(Series<double>);
   void add_feature(Series<string>);
   //append data
   void append(meta_element element);
   vector<string> get_all_features() { return feature_names; }; //return all feature names
   vector<string> get_all_numerical_features(string label_name); //return all numerical feature names
   void display_datas();                                        //print datas
   int get_data_numbers() { return totol_data_numbers; }        //return numbers of datas
   vector<single_data> numerical_dataset(string);               //get numerical dataset
};

void DataFrame::add_feature(Series<int> series)
{
   if (totol_data_numbers && series.data.size() != totol_data_numbers)
   {
      cout << "No matching for data numbers, Appending falied!";
   }
   if (!totol_data_numbers)
      totol_data_numbers = series.data.size();
   feature_names.push_back(series.feature_name);
   int_arrays.push_back(series);
   int pos = int_arrays.size() - 1;
   name_to_pos[series.feature_name] = pair<string, int>("int", pos);
}
void DataFrame::add_feature(Series<double> series)
{
   if (totol_data_numbers && series.data.size() != totol_data_numbers)
   {
      cout << "No matching for data numbers, Appending falied!";
   }
   if (!totol_data_numbers)
      totol_data_numbers = series.data.size();
   feature_names.push_back(series.feature_name);
   double_arrays.push_back(series);
   int pos = double_arrays.size() - 1;
   name_to_pos[series.feature_name] = pair<string, int>("double", pos);
}
void DataFrame::add_feature(Series<string> series)
{
   if (totol_data_numbers && series.data.size() != totol_data_numbers)
   {
      cout << "No matching for data numbers, Appending falied!";
   }
   if (!totol_data_numbers)
      totol_data_numbers = series.data.size();
   feature_names.push_back(series.feature_name);
   string_arrays.push_back(series);
   int pos = string_arrays.size() - 1;
   name_to_pos[series.feature_name] = pair<string, int>("string", pos);
}
void DataFrame::append(meta_element element)
{
   for (int i = 0; i < element.int_datas.size(); i++)
   {
      string feature = element.int_feature_names[i];
      if (!name_to_pos.count(feature))
      {
         cout << "inserting failed,no such feature!" << endl;
         return;
      }
      string type = name_to_pos[feature].first;
      if (type != "int")
      {
         cout << "the type of feature is not match!" << endl;
         return;
      }
   }
   for (int i = 0; i < element.double_datas.size(); i++)
   {
      string feature = element.double_feature_names[i];
      if (!name_to_pos.count(feature))
      {
         cout << "inserting failed,no such feature!" << endl;
         return;
      }
      string type = name_to_pos[feature].first;
      if (type != "double")
      {
         cout << "the type of feature is not match!" << endl;
         return;
      }
   }
   for (int i = 0; i < element.string_datas.size(); i++)
   {
      string feature = element.string_feature_names[i];
      if (!name_to_pos.count(feature))
      {
         cout << "inserting failed,no such feature!" << endl;
         return;
      }
      string type = name_to_pos[feature].first;
      if (type != "string")
      {
         cout << "the type of feature is not match!" << endl;
         return;
      }
   }
   if (element.int_datas.size() != int_arrays.size() || element.double_datas.size() != double_arrays.size() || element.string_datas.size() != string_arrays.size())
   {
      cout << "feature numbers not match!" << endl;
      return;
   }
   for (int i = 0; i < element.int_datas.size(); i++)
   {
      int pos = name_to_pos[element.int_feature_names[i]].second;
      int_arrays[pos].data.push_back(element.int_datas[i]);
   }
   for (int i = 0; i < element.double_datas.size(); i++)
   {
      int pos = name_to_pos[element.double_feature_names[i]].second;
      double_arrays[pos].data.push_back(element.double_datas[i]);
   }
   for (int i = 0; i < element.string_datas.size(); i++)
   {
      int pos = name_to_pos[element.string_feature_names[i]].second;
      string_arrays[pos].data.push_back(element.string_datas[i]);
   }
   totol_data_numbers++;
}
void DataFrame::display_datas()
{
   for (string &feature : feature_names)
   {
      string type = name_to_pos[feature].first;
      int pos = name_to_pos[feature].second;
      cout << left << setw(12) << feature << ":";
      if (type == "int")
      {
         for (int &x : int_arrays[pos].data)
         {
            cout << left << setw(12) << x << " ";
         }
         cout << endl;
      }
      if (type == "double")
      {
         for (double &x : double_arrays[pos].data)
         {
            cout << left << setw(12) << x << " ";
         }
         cout << endl;
      }
      if (type == "string")
      {
         for (string &x : string_arrays[pos].data)
         {
            cout << left << setw(12) << x << " ";
         }
         cout << endl;
      }
   }
}

// for splitting datas from csv file
vector<string> split(const string &str, const string &delim)
{
   vector<string> res;
   if ("" == str)
      return res;
   char *strs = new char[str.length() + 1];
   strcpy(strs, str.c_str());

   char *d = new char[delim.length() + 1];
   strcpy(d, delim.c_str());

   char *p = strtok(strs, d);
   while (p)
   {
      string s = p;
      res.push_back(s);
      p = strtok(NULL, d);
   }

   return res;
}

// judge the type of string
bool is_int(string str)
{
   if (str[0] == '0' && str.length() != 1)
      return false;
   for (int i = 0; i < str.length(); i++)
   {
      if (!(str[i] >= '0' && str[i] <= '9'))
         return false;
   }
   return true;
}
bool is_double(string str)
{
   int pos = 0;
   if (str.length() == 1)
      return false;
   if (str[0] == '0' && str[1] != '.')
      return false;
   for (int i = 1; i < str.length(); i++)
   {
      if (str[i] != '.' && (!(str[i] >= '0' && str[i] <= '9')))
         return false;
      if (str[i] == '.')
         pos = i;
   }
   if (!pos || pos == str.length() - 1)
      return false;
   else
      return true;
}

//convert Series<string> to Series<int>
Series<int> to_int(Series<string> str_series)
{
   Series<int> tmp;
   tmp.set_feature_name(str_series.get_feature_name());
   int len = str_series.size();
   for (int i = 0; i < len; i++)
   {
      stringstream stream;
      int t;
      stream << str_series[i];
      stream >> t;
      tmp.append(t);
   }
   return tmp;
}
//convert Series<string> to Series<double>
Series<double> to_double(Series<string> str_series)
{
   Series<double> tmp;
   tmp.set_feature_name(str_series.get_feature_name());
   int len = str_series.size();
   for (int i = 0; i < len; i++)
   {
      stringstream stream;
      double t;
      stream << str_series[i];
      stream >> t;
      tmp.append(t);
   }
   return tmp;
}
//string type series to one-hot serieses
vector<Series<int>> one_hot(Series<string> str_series)
{
   vector<Series<int>> Serieses;
   string outer_name = str_series.get_feature_name();
   vector<string> data = str_series.get_data();
   set<string> str_set;
   for (string str : data)
   {
      str_set.insert(str);
   }
   for (string str : str_set)
   {
      Series<int> tmp_series;
      string feature_name = outer_name + "_" + str;
      tmp_series.set_feature_name(feature_name);
      for (string strr : data)
      {
         if (strr == str)
            tmp_series.append(1);
         else
            tmp_series.append(0);
      }
      Serieses.push_back(tmp_series);
   }
   return Serieses;
}

// convert csv file into DataFrame
DataFrame read_csv(string path)
{

   fstream file(path, ios::in);
   string line;
   getline(file, line);
   vector<string> features = split(line, ",");
   vector<Series<string>> Serieses;
   for (auto x : features)
   {
      Series<string> tmp(x);
      Serieses.push_back(tmp);
   }
   int len = features.size();
   while (getline(file, line))
   {
      vector<string> elements = split(line, ",");
      for (int i = 0; i < len; i++)
      {
         Serieses[i].append(elements[i]);
      }
   }
   DataFrame df;
   for (auto feature : Serieses)
   {
      if (is_int(feature[0]))
      {
         //cout<<"int:"<<feature.get_feature_name()<<endl;
         Series<int> tmp_series = to_int(feature);
         df.add_feature(tmp_series);
      }
      else if (is_double(feature[0]))
      {
         //cout<<"double:"<<feature.get_feature_name()<<endl;
         Series<double> tmp_series = to_double(feature);
         df.add_feature(tmp_series);
      }
      else
      {
         df.add_feature(feature);
         vector<Series<int>> one_hot_serises = one_hot(feature);
         for (auto series : one_hot_serises)
         {
            df.add_feature(series);
         }
      }
   }
   return df;
}

// output numerical dataset for training
vector<single_data> DataFrame::numerical_dataset(string label_name)
{
   vector<single_data> dataset;
   for (int i = 0; i < totol_data_numbers; i++)
   {
      single_data tmp_data;
      for (Series<int> int_series : int_arrays)
      {
         if (int_series.feature_name == label_name)
         {
            tmp_data.label = int_series.data[i];
         }
         else
         {
            tmp_data.feature_to_value[int_series.feature_name] = double(int_series.data[i]);
         }
      }
      for (Series<double> double_series : double_arrays)
      {
         tmp_data.feature_to_value[double_series.feature_name] = double_series.data[i];
      }
      dataset.push_back(tmp_data);
      // cout<<tmp_data.label;
      // for(auto feature:tmp_data.feature_to_value)
      // {
      //    cout<<feature.first<<" "<<feature.second<<" ";
      // }
      //cout << endl;
   }
   return dataset;
}

//return all numerical feature names
vector<string> DataFrame::get_all_numerical_features(string label_name)
{
   vector<string> features;
   for (Series<int> int_series : int_arrays)
   {
      if(int_series.feature_name!=label_name)features.push_back(int_series.feature_name);
   }
   for (Series<double> double_series : double_arrays)
   {
      features.push_back(double_series.feature_name);
   }
   return features;   
}

#endif