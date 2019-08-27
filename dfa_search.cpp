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

void set_state(int prefix,int state, int *prefix2state,stack<int> &p2s_t,unordered_map<int,set< int>> &prefix_table, stack<int> &prefix_table_p, stack<set<int>> &prefix_table_t, bool trace, int &finish_prefix);
int update(int prefix,int state,int *prefix2state, stack<int> &p2s_t,unordered_map<int,set<int>> &different_group, unordered_map<int,set< int>> &prefix_table, stack<int> &prefix_table_p,stack<set<int>> &prefix_table_t, map< int,set< int>> &prefix2constrain,vector<vector<int>> &constrain_content, bool trace,int &finish_prefix);
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
	
	/* printing
	for(set<int>::iterator i=dmap[1][1].begin();i!=dmap[1][1].end(); i++)		cout<<"hi   "<<*i<<"   \n";	*/
	
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
	
	cout<<"found lower bound number of states "<< color<<"\n";

	bool fail = false;
	while(!fail){
		// searching
		int prefix_done=0;
		unordered_map< int,set< int>> prefix_table, state2prefix;
		//state starts with 1!!!!!!!!!!!!!!!!!!!
		int prefix2state[prefixes.size()];
		stack<set< int>> prefix_table_t;
		stack<int> prefix_table_p, p2s_t;
		// prefix and state, state prefix, prefix state for tuples. 
		int sb[color];
		for(int i=1;i<prefixes.size();i++){
			//symmetry breaking
			if(i<color+1)
				sb[i-1]=i;
			int j;
			//initalize
			if(different_group[0].find(i)==different_group[0].end())
				 j=1;
			else
				 j=2;
			while(j<=i && j<=color)
				prefix_table[i].insert(j++);
		}
		
		prefix2state[0]=1;	
		prefix_done++;
		int update_index=update(0,1,prefix2state,p2s_t,different_group,prefix_table,    prefix_table_p,prefix_table_t,prefix2constrain, constrain_content,false,prefix_done);
		
		//set_state(p0,s0,prefix2state,p2s_t,prefix_table,prefix_table_p,prefix_table_t,trace, prefix_done);
		
		//search
		//symmetry_breaking(color,*sb,*prefix2state,prefix_table,prefix_table_t, false);
		if(prefix_done==prefixes.size()){
			cout<<"final number of states "<<color<<"\n";
			break;
		}
		color++;
		break;
	}
	cout<<"end";
}

//search: 
//constrain_finding (most probable) via parser preper
//	insufficient info, assumption builder (use constrian to make assumptions) and update
// 	sufficient info, parser
//     all good, search
// 	   conflict, all possible values check, conflict diagonsis, backtrack, update assumption, continue.
//		


// strict deduction. should have no error. only error is in assumption

void set_state(int prefix,int state, int *prefix2state,stack<int> &p2s_t,unordered_map<int,set< int>> &prefix_table, stack<int> &prefix_table_p, stack<set<int>> &prefix_table_t, bool trace, int &finish_prefix){
	*(prefix2state+prefix)=state;	
	finish_prefix++;
	if(prefix_table[prefix].size()>1){
		if(trace){
			p2s_t.push(prefix);
			prefix_table_t.push(prefix_table[prefix]);
			prefix_table_p.push(prefix);
		}
		prefix_table[prefix].clear();
		prefix_table[prefix].insert(state);
	}
}
set<int> update(int prefix,int state,int *prefix2state, stack<int> &p2s_t,unordered_map<int,set<int>> &different_group, unordered_map<int,set< int>> &prefix_table, stack<int> &prefix_table_p,stack<set<int>> &prefix_table_t, map< int,set< int>> &prefix2constrain,vector<vector<int>> &constrain_content, bool trace,int &prefix_finish){
	set<int> different_set=different_group[prefix];
	int update_num;
	int update_index;
	int smallest=prefix_table[*(different_set.begin())].size();
	int smallestIndex=*(different_set.begin());
	for(auto i=different_set.begin();i!=different_set.end();i++){
		if(prefix_table[*i].size()<smallest){
			smallestIndex=*i;
			smallest=prefix_table[smallestIndex].size();
		}
		prefix_table[*i].erase(state);
		if(trace){
			prefix_table_t.push({state});
			prefix_table_p.push(*i);
		}
		int s = prefix_table[*i].size();
		if(s==0)
			return {-1};
		else if(s==1 && prefix2state[*i]==0){
			set_state(*i,*(prefix_table[*i].begin()),prefix2state,p2s_t,prefix_table,prefix_table_p,prefix_table_t,trace,prefix_finish);
			update_index=update(*i,*(prefix_table[*i].begin()),prefix2state,p2s_t,different_group,prefix_table,prefix_table_p,prefix_table_t,prefix2constrain, constrain_content,trace,prefix_finish);
			if(update_index==-1)
				return {-1};
			else{
				update_num=prefix_table[update_index].size();
				if(update_num<smallest){
					smallestIndex=update_index;
					smallest=update_num;
				}
			}
		}
	}
	
	set<int> p2c=prefix2constrain[prefix];
	for(auto k=p2c.begin();k!=p2c.end();k++){
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
			if(d1==0 || d2==0  ){
				if(d1>0||d2>0)
					undetermined=i-3;
				else
					undetermined=-1;
				continue;
			}
			if(op=-1)
				result+=(d1==d2);
			else
				result+=(d1!=d2);
		}
		if(undetermined==-2 && result>0)
			continue;
		// no undetermined and no result(all false) --> conflict
		if(undetermined==-2)
			return {-1};
		// exactly one undetermined
		if(undetermined>-1 && result==num_constrain/3-1){
			int op=content[undetermined++];
			int c1=content[undetermined++];
			int c2=content[undetermined++];
			int d1= *(prefix2state+c1);
			int d2= *(prefix2state+c2);
			int p0=(d1==0)?c1:c2;
			int s0=(d1!=0)?d1:d2;
			if(op==-1){
				set_state(p0,s0,prefix2state,p2s_t,prefix_table,prefix_table_p,prefix_table_t,trace,prefix_finish);
				update_index=update(p0,s0,prefix2state,p2s_t,different_group,prefix_table,prefix_table_p,prefix_table_t,prefix2constrain, constrain_content,trace,prefix_finish);
				if(update_index==-1)
					return {-1};
				else{
					update_num=prefix_table[update_index].size();
					if(update_num<smallest){
						smallestIndex=update_index;
						smallest=update_num;
					}
				}
			}
			else{
				prefix_table[p0].erase(s0);
				if(trace){
					prefix_table_t.push({s0});
					prefix_table_p.push(p0);
				}	
			}
		}
	}
	return smallestIndex;
}


int conflict(int prefix,int state,int *prefix2state, stack<int> &p2s_t,unordered_map<int,set<int>> &different_group, unordered_map<int,set< int>> &prefix_table, stack<int> &prefix_table_p,stack<set<int>> &prefix_table_t, map< int,set< int>> &prefix2constrain,vector<vector<int>> &constrain_content, bool trace,int &prefix_finish){
	
}
/*
void symmetry_breaking(int color,(int *)sb, (int *) prefix2state,unordered_map< int,set< int>> prefix_table, stack<tuple< int, int>> prefix_table_t,bool trace){
	for(int i=0;i<color;i++){
		if( prefix2state[*(sb+i)] < i+2 && prefix2[*(sb+i)]>0){
			
			for(int j=++*(sb+i);j<prefix_table.size();j++){
				int s = *(prefix_table[j].end()-1);
				prefix_table[j].erase(s);
				if(bool){
					prefix_table_t.push(make_tuple(j,s));
				}
			
			}
		}
	}
}

//access conflict by *((conflict+i*n)+j)
// appears that all constrains are in the form of or, so as long as one of them works, all of them works. 
int parser(vector<vector<int>> constrain_op,vector<vector<int>> constrain_content,int constrain,(int *)prefix2state ){
	int num_constrain = contrain_op[constrain].size();
	int result = 0;
	int j=0;
	for(int i=0;i<num_constrain;i++){
		char op = contrain_op[constrain][i];
		int a = constrain_content[cosntrain][j++];
		int b = constrain_content[cosntrain][j++];
		int c = *(prefix2state+a);
		int d = *(prefix2state+b);
		if(op='e')
			result=(c==d);
		else
			result=(c!=d);
		if(result==1)
			break;
	}
	return result;

}

//check whehter something is ready for parsing, return a set of integers that need assumptions
set<int> parse_preper()

//build assumptions. remember to take in tracer. also use a set of int as implication graph. void.
void assumption_builder()

// return conflict for database. use when parser return false.might want to look at dpll. 
vector<int> conflict_diagnosis()

//redo. take in tracer and relevent databases.
void backtracking()

*/
	/*
void merged(int state, int prefix, unordered_map<int,set<int>> prefix_table, bool track, stack<tuple<int,int>> prefix_table_trace){
	if(!track){
		
	}
	else{

	}
}
*/
	
