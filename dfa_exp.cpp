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
int color;
int set_state(int prefix,int state, int *prefix2state,stack<int> &p2s_t,unordered_map<int,set< int>> &prefix_table, stack<int> &prefix_table_p, stack<set<int>> &prefix_table_t, bool trace, int &finish_prefix);
int update(int prefix,int state,int *prefix2state, stack<int> &p2s_t,unordered_map<int,set<int>> &different_group, unordered_map<int,set< int>> &prefix_table, stack<int> &prefix_table_p,stack<set<int>> &prefix_table_t, map< int,set< int>> &prefix2constrain,vector<vector<int>> &constrain_content, bool trace,int &finish_prefix);
int search(int prefix,stack<int> &assumption_p,stack<int> &assumption_s, int &closest,int *prefix2state, stack<int> &p2s_t,unordered_map<int,set<int>> &different_group, unordered_map<int,set< int>> &prefix_table, stack<int> &prefix_table_p,stack<set<int>> &prefix_table_t, map< int,set< int>> &prefix2constrain,vector<vector<int>> &constrain_content, int &prefix_finish, int depth);
void printMap(unordered_map<int,set<int>> m);
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
	char file_path[] = "./dcts/dfa_4_try_1.dct";
	freopen(file_path,"r",stdin);
	int train_size,alsize;
	scanf("%d %d",&train_size,&alsize);
	string train_string[train_size];
	int train_label[train_size];
	for(int i = 0; i<train_size;i++ ){
		int label, string_length;
		scanf("%d %d",&label,&string_length);
		string s;
		int count =0;
		while(count<string_length){
			int d;
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
	unordered_map<int,set<int>> different_group;


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
	//buildin txt file for coloring
	/*
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
	*/
	
	color=2;
	cout<<"found lower bound number of states "<< color<<"\n";
	
	bool fail = false;
	while(!fail){
		cout<<"****************************************\n";
		cout<<"****************************************\n";
		cout<<"****************************************\n";
		cout<<"start color "<<color<<"\n";

		// searching
		int prefix_finish=0;
		unordered_map< int,set< int>> prefix_table;
		//state starts with 0!!!!!!!!!!!!!!!!!!! unlabel is -3
		// color starts at prefixes.size()+color_num+1
		int prefix2state[prefixes.size()+color];
		for(int i=0;i<prefixes.size();i++)
			prefix2state[i]=-3;
		for(int i=prefixes.size();i<prefixes.size()+color;i++){
			prefix2state[i]=i-prefixes.size();
		}
		stack<set< int>> prefix_table_t;
		stack<int> prefix_table_p, p2s_t;
		// prefix and state, state prefix, prefix state for tuples. 
		int sb[color];
		prefix_table[0].insert(0);
		prefix2state[0]=0;
		prefix_finish++;
		for(int i=1;i<prefixes.size();i++){
			//symmetry breaking
			if(i<color+1)
				sb[i-1]=i;
			int j;
			//initalize
			if(different_group[0].find(i)==different_group[0].end()){
				j=0;
			}
			else{
				 j=1;
			}
			while(j<=i && j<color)
				prefix_table[i].insert(j++);
		}
		
		
		int update_index=update(0,0,prefix2state,p2s_t,different_group,prefix_table,    prefix_table_p,prefix_table_t,prefix2constrain, constrain_content,false,prefix_finish);
		cout<<prefix_table[1].size()<<" !!!!!! \n";
		int closest=1;
		stack<int> assumption_p, assumption_s;

		if(update_index<0){
			cout<<"error\n";
		}
		else{
			cout<<"start search\n";
		//set_state(p0,s0,prefix2state,p2s_t,prefix_table,prefix_table_p,prefix_table_t,trace, prefix_done);
			int s = search(update_index,assumption_p,assumption_s, closest,prefix2state, p2s_t,different_group, prefix_table, prefix_table_p,prefix_table_t, prefix2constrain,constrain_content, prefix_finish, 0);
		//search
		//symmetry_breaking(color,*sb,*prefix2state,prefix_table,prefix_table_t, false);
			if(s){
				cout<<"final number of states "<<color<<"\n";
				break;
			}
		}
		color++;
		if(color>15){
			cout<<"faileddddddddddddd!!!!!!!!!!!!!!";
			break;
		}
	}
	cout<<"end";
}
//the tuple vs pair debate too!
int search(int prefix_candidate,stack<int> &assumption_p,stack<int> &assumption_s, int &closest,int *prefix2state, stack<int> &p2s_t,unordered_map<int,set<int>> &different_group, unordered_map<int,set< int>> &prefix_table, stack<int> &prefix_table_p,stack<set<int>> &prefix_table_t, map< int,set< int>> &prefix2constrain,vector<vector<int>> &constrain_content, int &prefix_finish, int depth){
	cout<<"##################################################\n";
	cout<<"##################################################\n";
	cout<<"start searching for color" <<color<<"\n";
	cout<<"the depth is "<<depth<<"\n";
	cout<<"number of prefix finish is "<<prefix_finish<<" out of "<<prefix_table.size()<<"\n";
	if(prefix_finish==prefix_table.size())
		return 1;
	int search_result, prefix;
	int progress=0;
	int small=-3;
	int smallS=-100;
	int tmp;
	for(int i=0;i<prefix_table.size();i++){
		tmp=prefix_table[i].size();
		if(prefix_table[i].size()==1)
			progress++;
		if(small==-3 && tmp>1 && *(prefix2state+i)==-3){
			small=i;
			smallS=tmp;
		}
		else if(small!=-3 && tmp>1 && tmp<smallS&&*(prefix2state+i)==-3){
			small=i;
			smallS=tmp;
		}
		
	}
	if(prefix2state[small]!=-3)
		cout<<"@@@@@@@@@@@@@@ problem @@@@@@@@\n";
	if(progress==prefix_table.size())
		return 1;
	prefix=small;
	cout<<"!!!!!!! now the search is for prfix "<<prefix<<"\n";
	cout<<"progress is "<<progress<<"out of "<<prefix_table.size()<<"\n";
	int assumption_p_s=assumption_p.size();
	int p2s_t_s=p2s_t.size();
	int pt_p_s=prefix_table_p.size();
	int p_f=prefix_finish;
	set<int> pt=prefix_table[prefix];
	int count=0;
	for(auto i=pt.begin();i!=pt.end();i++){
		
		cout<<"searching process for prefix "<<prefix<<"using state "<<*i<< " with count "<<count<<" out of "<<pt.size()<<"\n";
		count++;
		assumption_p.push(prefix);
		assumption_s.push(*i);
		set_state(prefix,*i,prefix2state,p2s_t,prefix_table,prefix_table_p,prefix_table_t,true,prefix_finish);
		int update_index=update(prefix,*i,prefix2state,p2s_t,different_group,prefix_table,prefix_table_p,prefix_table_t,prefix2constrain, constrain_content,true,prefix_finish);
		while(update_index>-1){
			cout<<"searching assumption works for prefixs "<<prefix<<"with state "<<*i<<" continuing next search\n";
			search_result=search(update_index,assumption_p,assumption_s,++closest,prefix2state,p2s_t,different_group,prefix_table,prefix_table_p,prefix_table_t,prefix2constrain, constrain_content,prefix_finish,depth+1);
			if(search_result){
				//cout<<"search success, returning to initial depth\n";
				return 1;
			}
			else if(search_result==0 && depth == 0){
				cout<<"search failed. some prefixes have no choice. Returned to highest decision level for redo. Prefix "<<prefix<<"\n";
				prefix_finish=p_f;
				while(pt_p_s<prefix_table_p.size()){
					set<int> ptset=prefix_table_t.top();
					int ptint=prefix_table_p.top();
					prefix_table_t.pop();
					prefix_table_p.pop();
					prefix_table[ptint].insert(ptset.begin(),ptset.end());	
				}
				vector<int> c;
				while(assumption_p_s<assumption_p.size()){
					int a_p=assumption_p.top();
					int a_s=assumption_s.top()+prefix_table.size();
					assumption_p.pop();
					assumption_s.pop();
					c.push_back(-2);				
					c.push_back(a_p);
					c.push_back(a_s);
				}
				constrain_content.push_back(c); //use prefix_table.size()+color_num (>=1) for constrain_content
				cout<<"redoing update and search\n";
				update_index=update(prefix,*i,prefix2state,p2s_t,different_group,prefix_table,prefix_table_p,prefix_table_t,prefix2constrain, constrain_content,true,prefix_finish);
			}
		
			else{
				cout<<"passing up failure for prefix "<<prefix<<"\n";
				return 0;
			}	
		}
		cout<<"failed while loop. change assumption for the next state from "<<*i<<" for prefix "<<prefix<<"\n";
		prefix_finish=p_f;
		cout<<"stop 1\n";
		while(pt_p_s<prefix_table_p.size()){
			cout<<"stop 2\n";
			set<int> ptset=prefix_table_t.top();
			int ptint=prefix_table_p.top();
			prefix_table_t.pop();
			prefix_table_p.pop();
			prefix_table[ptint].insert(ptset.begin(),ptset.end());	
		}

		cout<<"stop 3\n";
		assumption_p.pop();
		assumption_s.pop();
	}
	///home/krong/Documents/gdb_init
	cout<<"all assumptions for prefix "<<prefix<<" failed. going upward for redo\n";
	return 0;
}


// strict deduction. should have no error. only error is in assumption
int set_state(int prefix,int state, int *prefix2state,stack<int> &p2s_t,unordered_map<int,set< int>> &prefix_table, stack<int> &prefix_table_p, stack<set<int>> &prefix_table_t, bool trace, int &finish_prefix){
	if(prefix_table[prefix].find(state)==prefix_table[prefix].end()){
		cout<<"non existence";
		return -1;
	}
	if( *(prefix2state+prefix)==-3){
		finish_prefix++;
	}
	*(prefix2state+prefix)=state;	
	cout<<"setting state for prefix "<<prefix<<" wtih state "<<state<<"\n";
	if(prefix_table[prefix].size()>1){
		if(trace){
			p2s_t.push(prefix);
			prefix_table_t.push(prefix_table[prefix]);
			prefix_table_p.push(prefix);
		}
		prefix_table[prefix].clear();
		prefix_table[prefix].insert(state);
	}
	return 1;
}
int update(int prefix,int state,int *prefix2state, stack<int> &p2s_t,unordered_map<int,set<int>> &different_group, unordered_map<int,set< int>> &prefix_table, stack<int> &prefix_table_p,stack<set<int>> &prefix_table_t, map< int,set< int>> &prefix2constrain,vector<vector<int>> &constrain_content, bool trace,int &prefix_finish){
	cout<<"##################################################\n";
	cout<<"starting updating: updating for prefix "<<prefix<<"with color "<<color<<"\n";
	set<int> different_set=different_group[prefix];
	int update_num;
	int update_index;
	int smallest=0;
	int smallestIndex=1;
	/*
	while(smallest<2 && prefix2state[smallestIndex]!=-3){
		smallestIndex++;
		smallest=prefix_table[smallest].size();
		if(smallestIndex>prefix_table.size())
			smallestIndex=1;

	}
	*/
	set<int> p2c=prefix2constrain[prefix];

	cout<<"starting constrain evaluation\n";
	int test_v=0;	
	
	for(auto k=p2c.begin();k!=p2c.end();k++){
		if(test_v>1 && test_v%1000==0)
		cout<<"update's constrain evaluation for prefix"<< prefix<<"using constrain "<<*k<< " count " <<test_v<< "out of " <<p2c.size()<<"\n";
		test_v++;
		int constrain=*k;
		vector<int> content=constrain_content[constrain];
		int num_constrain = content.size();
		int undetermined=-2;
		int result=0;
		for(int i=0;i<num_constrain;){
			int op=content[i++];
			int c1=content[i++];
			int c2=content[i++];
			int d1= *(prefix2state+c1);
			int d2= *(prefix2state+c2);
			if(d1==-3 || d2==-3  ){
				if(d1>-3||d2>-3)
					undetermined=i-3;
				else
					undetermined=-1;
				continue;
			}
			if(op==-1)
				result+=(d1==d2);
			else
				result+=(d1!=d2);
		}
		//cout<<"outcome: undetermined "<< undetermined<<" result "<<result<<" \n";
		if(undetermined==-2 && result>0)
			continue;
		// no undetermined and no result(all false) --> conflict
		if(undetermined==-2){
			cout<<"contradict constrains\n";
			return -1;
		}
		// exactly one undetermined
		if(undetermined>-1 && result>0 && result==num_constrain/3-2){
			int op=content[undetermined++];
			int c1=content[undetermined++];
			int c2=content[undetermined++];
			int d1= *(prefix2state+c1);
			int d2= *(prefix2state+c2);
			int p0=(d1==-3)?c1:c2;
			int s0=(d1!=-3)?d1:d2;
			if(op==-1 && *(prefix2state+p0)==-3){
				if(p0==1){
					cout<<"p0 "<<p0<<" s0 "<<s0<<" c1 "<<c1<<" c2 "<<c2<<" d1 "<<d1<<" d2 "<<d2<<"\n";
				}
				cout<<"forced equality for prefix "<<p0<<" with state "<<s0<<"\n";
				int set_state_index=set_state(p0,s0,prefix2state,p2s_t,prefix_table,prefix_table_p,prefix_table_t,trace,prefix_finish);
				if(p0==1){
					cout<<"did this update??? " <<prefix2state[1]<<"\n";
				}
				if(set_state_index==-1){
					cout<<"failed forced equality, returning -1 \n";
					return -1;

				}
				update_index=update(p0,s0,prefix2state,p2s_t,different_group,prefix_table,prefix_table_p,prefix_table_t,prefix2constrain, constrain_content,trace,prefix_finish);
				if(update_index==-1){
					cout<<"failed forced equality, returning -1 \n";
					return -1;
				}
				else{
					update_num=prefix_table[update_index].size();
					if(update_num<smallest and update_num>1){
						smallestIndex=update_index;
						smallest=update_num;
					}
				}
			}
			else{
				cout<<"forced inequality for prefix "<<p0<<" with state "<<s0<<"\n";
				prefix_table[p0].erase(s0);
				if(trace){
					prefix_table_t.push({s0});
					prefix_table_p.push(p0);
				if(prefix_table[p0].size()==1 && prefix2state[p0]==-3){
					cout<<"success forced inequality. updating "<<p0<<"\n";
					set_state(p0,*(prefix_table[p0].begin()),prefix2state,p2s_t,prefix_table,prefix_table_p,prefix_table_t,trace,prefix_finish);
					update_index=update(p0,*(prefix_table[p0].begin()),prefix2state,p2s_t,different_group,prefix_table,prefix_table_p,prefix_table_t,prefix2constrain, constrain_content,trace,prefix_finish);
				}
				if(update_index=-1){
					cout<<"failed force inequality update, returning -1 \n";
					return -1;
			}
				
				if(prefix_table[p0].size()==0)
					cout<<"failed force inequality update due to size==0, returning -1 \n";
					return -1;
				}	
			}
		}
	}
	cout<<" end of constrain eval\n";
	int count=0;

	for(auto i=different_set.begin();i!=different_set.end();i++){
		if(count%1000==0 && count>1)
		cout<<"update's difference evaluation for prefix"<< prefix<<"with state "<<*i<<" count " <<count<< "out of " <<different_set.size()<<"\n";
		count++;
		if(prefix_table[*i].size()<smallest and prefix_table[*i].size()>1){
			smallestIndex=*i;
			smallest=prefix_table[smallestIndex].size();
		}
		prefix_table[*i].erase(state);
		if(trace){
			prefix_table_t.push({state});
			prefix_table_p.push(*i);
		}
		int s = prefix_table[*i].size();
		if(s==0){
			cout<<"failed difference, returning -1 \n";
			return -1;
		}
		else if(s==1 && prefix2state[*i]==-3){
			cout<<"success difference. updating "<<*i<<"\n";
			set_state(*i,*(prefix_table[*i].begin()),prefix2state,p2s_t,prefix_table,prefix_table_p,prefix_table_t,trace,prefix_finish);
			update_index=update(*i,*(prefix_table[*i].begin()),prefix2state,p2s_t,different_group,prefix_table,prefix_table_p,prefix_table_t,prefix2constrain, constrain_content,trace,prefix_finish);
			if(update_index==-1){
				cout<<"failed difference update, returning -1 \n";
				return -1;
			}
			else{
				update_num=prefix_table[update_index].size();
				if(update_num<smallest and update_num>1){
					smallestIndex=update_index;
					smallest=update_num;
				}
			}
		}
	}

	cout<<"end of different set eval\n";
	return smallest;
}

void printMap(unordered_map<int,set<int>> m, int n){
	for(auto i=m[n].begin();i!=m[n].end();i++){
		cout<<*i<<"  ";
	}
	cout<<"\n";
}

