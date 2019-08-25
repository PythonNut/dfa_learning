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
int main(){
	//might want to redo to file opener.
	
	//read inputs from file. might want to optimize to take prefix set and suffix set in here.
	char file_path[] = "./dcts/dfa_12_try_7.dct";
	freopen(file_path,"r",stdin);
	unsigned int train_size,alsize;
	scanf("%d %d",&train_size,&alsize);
	string train_string[train_size];
	unsigned int train_label[train_size];
	for(unsigned int i = 0; i<train_size;i++ ){
		unsigned int label, string_length;
		scanf("%d %d",&label,&string_length);
		string s;
		unsigned int count =0;
		while(count<string_length){
			unsigned int d;
			scanf(" %d",&d);
			s=s+to_string(d);
			count++;
			
		}
		train_string[i] = s;
		train_label[i]=label;
	}

	//enumerate fixes. generate the prefix and suffix sets
	set<string, decltype(&cmp)> prefixes(&cmp);
	set<string>  suffixes;
	string prefix, suffix;
	for(unsigned int i=0;i<train_size; i++){
		for(unsigned int j=0;j< train_string[i].length()+1; j++){
			prefix=train_string[i].substr(0,j);
			suffix=train_string[i].substr(j,train_string[i].length());
			prefixes.insert(prefix);
			suffixes.insert(suffix);
		}
	}
	
	// build dictionaries
	unordered_map<string,unsigned int> prefixes_map, suffixes_map;
	unordered_map<unsigned int,string>prefixes_indexes;
	unsigned int count=0;
	for(set<string>::iterator i=prefixes.begin();i!=prefixes.end(); i++){
		prefixes_indexes[count]=*i;
		prefixes_map[*i]=count++;
	}
	count=0;
	for(set<string>::iterator i=suffixes.begin();i!=suffixes.end(); i++){
		suffixes_map[*i]=count++;
	}
	
	/*
	//printing
	for(set<string>::iterator i=prefixes.begin();i!=prefixes.end(); i++){
		cout<<prefixes_map[*i]<<"   "<<*i<<"   \n";
	}
	cout<<"------\n";
	for(set<string>::iterator i=suffixes.begin();i!=suffixes.end(); i++){
		cout<<suffixes_map[*i]<<"   "<<*i<<"   \n";
	}
	*/


	// include unstated and stated. use prefix id. 
	// the range is 0-prefix_size-1, prefix_size-state-1+prefix_size 
	// dictionary: 0 is unstated, 1 is stated
	// set includes possible number of states
	unordered_map<unsigned int,set<unsigned int>> different_group, constrain2prefix,prefix2constrain;
	unordered_map<unsigned int,queue<char>> constrains;	


	// build distinguishability map
	unordered_map<unsigned int,unordered_map<unsigned int,set<unsigned int>>> dmap;
	unordered_map<unsigned int,unsigned int> map_label;
	for(unsigned int i=0;i<train_size; i++){
		for(unsigned int j=0;j< train_string[i].length()+1; j++){
			prefix=train_string[i].substr(0,j);
			suffix=train_string[i].substr(j,train_string[i].length());
			unsigned int pi = prefixes_map[prefix];
			unsigned int si = suffixes_map[suffix];
			
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
	unordered_map<unsigned int,set<tuple<unsigned int,unsigned int>>> pre_dmap;
	unordered_map<string,unsigned int> suffixes_map2;

	unordered_map<unsigned int,string> suffixes_map20;
	unsigned int k=0;
	unsigned int r=0;
	for(auto i=prefixes.begin();i!=prefixes.end(); i++){
		for(unsigned int j=0;j< (*i).size()+1; j++){
			prefix=(*i).substr(0,j);
			suffix=(*i).substr(j,(*i).size());
			if(prefixes_map.find(prefix)!=prefixes_map.end()){
				cout<<"IN!  "<<prefix<<"  "<<*i<<"\n";
				unsigned int pi = prefixes_map[prefix];	
				if(suffixes_map2.find(suffix)==suffixes_map2.end()){
					suffixes_map20[k]=suffix;
					suffixes_map2[suffix]=k++;
				}
				unsigned int si = suffixes_map2[suffix];
				tuple<unsigned int,unsigned int> t=make_tuple(r,pi);
				pre_dmap[si].insert(t);
			}
			else
				continue;
			
			}
		r++;
	}
		
	for(unordered_map<unsigned int,set<tuple<unsigned int,unsigned int>>>::iterator i=pre_dmap.begin();i!=pre_dmap.end(); i++){
		
		//cout<<"yesss   "<<suffixes_map20[i->first]<<"\n";
		for(auto j=i->second.begin();j!=i->second.end();j++){
			//cout<<get<0>(*j)<<"  no "<<prefixes_indexes[get<0>(*j)]<<" hi  "<<prefixes_indexes[get<1>(*j)]<<"   \n";
			
			

		}
	}
	
	/* printing
	for(set<int>::iterator i=dmap[1][1].begin();i!=dmap[1][1].end(); i++){
		
		cout<<"hi   "<<*i<<"   \n";
	}
	*/
	//conflict graph
	unsigned int edgeNum=0;
	stack<unsigned int> edge_0,edge_1;
	for(auto i=dmap.begin();i!=dmap.end();i++){
		unordered_map<unsigned int,set<unsigned int>> sec = i->second;
		if(sec.find(0)!=sec.end() && sec.find(1) != sec.end()){
			for(set<unsigned int>::iterator j=sec[0].begin(); j!=sec[0].end(); j++){
				for(set<unsigned int>::iterator k=sec[1].begin();k!=sec[1].end();k++){
					different_group[*j].insert(*k);					
					different_group[*k].insert(*j);					
					edge_0.push(*j);
					edge_1.push(*k);
					edgeNum++;
				}
			}	
		}
	}

	/*  printing
	for(set<int>::iterator i=different_group[0].begin();i!=different_group[0].end(); i++){
		
		cout<<"hi   "<<*i<<"   \n";
	}
	*/
	//buildin txt file for coloring
	freopen("output.txt","w",stdout);
	ofstream dimacs;
  	dimacs.open ("search_dimacs.col");
	dimacs<<"p edge "<<prefixes.size()<<" "<<edgeNum<<"\n";
	while(!edge_0.empty()){
		dimacs<<"e "<<edge_0.top()+1<<" "<<edge_1.top()+1<<"\n";
		edge_0.pop();
		edge_1.pop();
	}
	system("./../fastColor/fastColor -f search_dimacs.col -t 0");	
	freopen("output.txt","r",stdin);
	int color;
	scanf("%*[^\n]\n");
	scanf("%*[^\n]\n");
	scanf("%*f %*s %*s %*s %d",&color); 
	cout<<color<<"\n";
	


	
	bool fail = true;
	while(!fail){
		// searching
		unordered_map<unsigned int,set<unsigned int>> prefix_table, state2prefix;
		//do we need state2prefix????
		unsigned int prefix2state[prefixes.size()];
		stack<tuple<unsigned int,unsigned int>> prefix_table_trace, state2prefix_trace, prefix2state_trace;
		
		for(unsigned int i=1;i<prefixes.size();i++){
			
			unsigned int j=0;
			while(j<i || j<color);
				prefix_table[i].insert(j++);
		}
		prefix2state[0]=0;
		state2prefix[0].insert(0);
		//merged(0,0,prefix_table,false,prefix_table_trace);
		//search

		color++;
	}
	cout<<"end";
}

//access conflict by *((conflict+i*n)+j)

void parser(){
	string constrain = 
	for
}

	/*
void merged(int state, int prefix, unordered_map<int,set<int>> prefix_table, bool track, stack<tuple<int,int>> prefix_table_trace){
	if(!track){
		
	}
	else{

	}
}
*/
	
