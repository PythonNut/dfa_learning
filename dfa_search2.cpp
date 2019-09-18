#include<stack>
#include<tuple>
#include<map>
#include<cstdio>
#include<iostream>
#include<set>
#include<algorithm>
#include<cstdlib>
#include<tuple>
#include<string>
#include<stdlib.h>
#include"mcqd.h"
#include <fstream>
#include <bits/stdc++.h>
using namespace std;

// varaible namings and function definitions
int numState; // previously "color"
unordered_map<int, set<int>> different_group;
vector<int> prefix2state;
struct change{
	int state; 
	set<int> candidates;
	bool p2s_change; //true --> change prefix2state by assigning a state id
};
stack<change> changes;
unordered_map<int,set<int>> prefix_table;
map<int,set<int>> prefix2constrain;
vector<vector<int>> constrain_content; 

void process();
int set_state(int prefix, int state, bool trace);
int update(int prefix, int state, bool trace);
int search(int prefix, stack<int> &assumption_prefix, stack<int> &assumption_state,int depth);
bool cmp(string a, string b);

// comparator helper function
bool cmp(string a, string b) {
    if(a.size()!=b.size())
        return a.size()<b.size();
    else{
        char aa[a.size()];
        char bb[b.size()];
        strcpy(aa,a.c_str());
        strcpy(bb,b.c_str());
        return lexicographical_compare(aa,aa+a.size(),bb,bb+b.size());
    }
}

void process(){
    // READ IN FILE
	int train_size,alsize;
	FILE * pFile;
	
	pFile = fopen("./dcts/dfa_12_try_1.dct", "r");
    fscanf(pFile,"%d %d",&train_size,&alsize);
    printf("12 states\n");
	string train_string[train_size];
	int train_label[train_size];

    for(int i = 0; i<train_size;i++ ){
        int label, string_length;
        fscanf(pFile,"%d %d",&label,&string_length);
        string s;
        int count =0;
        while(count<string_length){
            int d;
            fscanf(pFile," %d",&d);
            s=s+to_string(d);
            count++;
        }
        train_string[i] = s;
        train_label[i]=label;
    }
	
	printf("finish reading file \n");
	cout<<train_string[0]<<" "<<train_string[3]<<"\n";

	//enumerate pre/suffix. generate the prefix and suffix sets
    set<string, decltype(&cmp)> prefixes(&cmp);
    set<string>  suffixes;
    string prefix, suffix;
    for(int i=0;i<train_size; i++){
        for(int j=0;j< train_string[i].length()+1; j++){
            prefix=train_string[i].substr(0,j);
            suffix=train_string[i].substr(j,train_string[i].length());
            prefixes.insert(prefix);
            suffixes.insert(suffix);
        }
    }

	// build dictionaries
    unordered_map<string,int> prefixes_map, suffixes_map;
    unordered_map<int,string>prefixes_indexes;
    int count=0;
    for(set<string>::iterator i=prefixes.begin();i!=prefixes.end(); i++){
        prefixes_indexes[count]=*i;
        prefixes_map[*i]=count++;
    }
    count=0;
    for(set<string>::iterator i=suffixes.begin();i!=suffixes.end(); i++){
        suffixes_map[*i]=count++;
    }
	
    // build distinguishability map
    unordered_map<int,unordered_map<int,set<int>>> dmap;
    unordered_map<int,int> map_label;
    for(int i=0;i<train_size; i++){
        for(int j=0;j< train_string[i].length()+1; j++){
            prefix=train_string[i].substr(0,j);
            suffix=train_string[i].substr(j,train_string[i].length());
            int pi = prefixes_map[prefix];
            int si = suffixes_map[suffix];

            if(j==train_string[i].length()+1){
                map_label[pi]=train_label[i];
            }
            else{
                map_label[pi]=2;
            }

            dmap[si][train_label[i]].insert(pi);
            }
    }
   //for condition 2
    unordered_map<int,set<tuple<int,int>>> pre_dmap;
    //long prefix id, short prefix id
    unordered_map<string,int> suffixes_map2;
    unordered_map<int,string> suffixes_map20;
    int suf=0;
    int long_pre=0;
    for(set<string>::iterator i=prefixes.begin();i!=prefixes.end(); i++){
        for( int j=0;j< (*i).size(); j++){
            prefix=(*i).substr(0,j);
            suffix=(*i).substr(j,(*i).size());
            if(prefixes_map.find(prefix)!=prefixes_map.end()){
                int pi = prefixes_map[prefix];
                if(suffixes_map2.find(suffix)==suffixes_map2.end()){
                    suffixes_map20[suf]=suffix;
                    suffixes_map2[suffix]=suf++;
                }
                int si = suffixes_map2[suffix];
                pre_dmap[si].insert(make_tuple(long_pre,pi));
            }
            else
                continue;

        }
        long_pre++;
    }
    printf("finish condition 2 prep\n");

    map<int,set<int>> prefix2constrain; //constrain2prefix;
    // cannot change the above to vector bc we need it to be indexed properly
    vector<vector<int>>constrain_content;
    for(unordered_map< int,set<tuple< int, int>>>::iterator i=pre_dmap.begin();i!=pre_dmap.end(); i++){

        //cout<<"yesss   "<<suffixes_map20[i->first]<<"\n";
        for(auto j=i->second.begin();j!=i->second.end();j++){
            //cout<<get<0>(*j)<<"  no "<<prefixes_indexes[get<0>(*j)]<<" hi  "<<prefixes_indexes[get<1>(*j)]<<"   \n";
            auto k2=j;
            for(auto k=k2++;k!=i->second.end();k++){
                vector<int> c;
                int c_id=constrain_content.size();
                int long_1=get<0>(*j);
                int long_2=get<0>(*k);
                int short_1=get<1>(*j);
                int short_2=get<1>(*k);
                if(long_1!=long_2 ||short_1!=short_2){
                    c.push_back(-2);
                    c.push_back(short_1);
                    c.push_back(short_2);
                    c.push_back(-1);
                    c.push_back(long_1);
                    c.push_back(long_2);
                    constrain_content.push_back(c);
                    //constrain2prefix[c_id]={short_1,short_2,long_1,long_2};
                    prefix2constrain[short_1].insert(c_id);
                    prefix2constrain[short_2].insert(c_id);
                    prefix2constrain[long_1].insert(c_id);
                    prefix2constrain[long_2].insert(c_id);
                }
            }
        }
    }
    printf("finish condition 2\n");
	//conflict graph
    int edgeNum=0;
    stack< int> edge_0,edge_1;
    for(auto i=dmap.begin();i!=dmap.end();i++){
        unordered_map< int,set< int>> sec = i->second;
        if(sec.find(0)!=sec.end() && sec.find(1) != sec.end()){
            for(set< int>::iterator j=sec[0].begin(); j!=sec[0].end(); j++){
                for(set< int>::iterator k=sec[1].begin();k!=sec[1].end();k++){
                    different_group[*j].insert(*k);
                    different_group[*k].insert(*j);
                    edge_0.push(*j);
                    edge_1.push(*k);
                    edgeNum++;
                }
            }
        }
    }
    cout<<"finish conflict graph output\n";

}

int set_state(int prefix, int state, bool trace){
}
int update(int prefix, int state, bool trace){
}
int search(int prefix, stack<int> &assumption_prefix, stack<int> &assumption_state,int depth){
}	
int main(){
	process();
}
