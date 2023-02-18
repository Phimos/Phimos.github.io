---
title: "2018大数据研究中心夏令营上机考试题解"
date: "2020-07-07 14:54:18"
tags: ["Algorithm", "Interview"]
---

*因为就是准备笔试时候自己刷了刷题，所以基本都只在本地跑过了样例*

## A. 第n小的质数

`Implementation`

```c++
#include<bits/stdc++.h>
using namespace std;

#define maxn 200010

bool isprime[maxn];
vector<int> primes;

void solve(){
    fill(isprime, isprime+maxn, true);
    int k=2;
    while(k<maxn-1){
        for(int i=2*k;i<maxn-1;i+=k){
            isprime[i] = false;
        }
        for(++k;!isprime[k];++k){
            ;
        }
    }
    for(int i=2;i<maxn;++i){
        if(isprime[i]){
            primes.push_back(i);
        }
    }
}


int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    solve();

    int n;
    cin>>n;
    cout<<primes[n-1]<<endl;
}
```



## B. 潜伏者

`Implementation`

```c++
#include<bits/stdc++.h>
using namespace std;

map<char, char> trans;

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    string s, t;
    cin>>s>>t;
    bool failed = false;
    for(int i=0;i<s.length();++i){
        if(trans.find(s[i]) != trans.end() && trans[s[i]] != t[i]){
            failed = true;
            break;
        }
        else{
            trans[s[i]] = t[i];
        }
    }

    for(int i=0;i<26;++i){
        if(trans.find('A'+i) == trans.end()){
            failed = true;
            break;
        }
    }
    
    cin>>s;
    if(failed){
        cout<<"Failed"<<endl;
    }
    else{
        for(int i=0;i<s.length();++i){
            cout<<trans[s[i]];
        }
        cout<<endl;
    }

}
```



## C. 逃离迷宫

`BFS`

```c++
#include<bits/stdc++.h>
using namespace std;

int m, t;
char mat[12][12];
int cnt[12][12];
int sx, sy, ex, ey;
int dir[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int k;
    cin>>k;
    while(k--){
        cin>>m>>t;
        for(int i=0;i<m;++i){
            for(int j=0;j<m;++j){
                cin>>mat[i][j];
                if(mat[i][j] == 'S'){
                    sx = i;
                    sy = j;
                }
                if(mat[i][j] == 'E'){
                    ex = i;
                    ey = j;
                }
            }
        }

        queue<pair<int, int>> q;
        for(int i=0;i<m;++i){
            for(int j=0;j<m;++j){
                cnt[i][j] = -1;
            }
        }
        q.push(make_pair(sx, sy));
        cnt[sx][sy] = 0;
        while(!q.empty()){
            int x = q.front().first;
            int y = q.front().second;
            q.pop();

            for(int i=0;i<4;++i){
                int nx = x + dir[i][0];
                int ny = y + dir[i][1];
                if(0 <= nx && nx < m && 0 <= ny && ny < m && (mat[nx][ny] == '.' || mat[nx][ny] == 'E') && cnt[nx][ny] == -1){
                    cnt[nx][ny] = cnt[x][y] + 1;
                    q.push(make_pair(nx, ny));
                }
            }
        }

        if(cnt[ex][ey] == -1 || cnt[ex][ey] > t){
            cout<<"NO"<<endl;
        }
        else{
            cout<<"YES"<<endl;
        }
    }
}
```



## D. 跑步

`DP`

```c++
#include<bits/stdc++.h>
using namespace std;

#define maxn 10010
#define maxm 502

int n, m;
int d[maxn];
int dp[maxn][maxm];

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    cin>>n>>m;
    for(int i=0;i<n;++i){
        cin>>d[i];
    }

    for(int i=0;i<n;++i){
        for(int j=0;j<m;++j){
            dp[i][j] = -1;
        }
    }

    dp[0][0] = 0;

    for(int i=0;i<n;++i){
        for(int j=0;j<=m;++j){
            if(dp[i][j] != -1){
                if(j<m){
                    dp[i+1][j+1] = max(dp[i+1][j+1], dp[i][j] + d[i]);
                }
                if(i+j <= n){
                    dp[i+j][0] = max(dp[i+j][0], dp[i][j]);
                }
                if(j == 0){
                    dp[i+1][0] = max(dp[i+1][0], dp[i][0]);
                }
            }
        }
    }
    
    cout<<dp[n][0]<<endl;
}
```



## E. What time is it?

`Implementation`

```c++
#include<bits/stdc++.h>
using namespace std;

string s[3];

bool check_num(int num, int idx){
    switch (num)
    {
    case 0:
        return s[1][idx+1]==' ';
    case 1:
        return s[0][idx+1]==' '&&s[1][idx]==' '&&s[1][idx+1]==' '&&s[2][idx]==' '&&s[2][idx+1]==' ';
    case 2:
        return s[1][idx]==' '&&s[2][idx+2]==' ';
    case 3:
        return s[1][idx]==' '&&s[2][idx]==' ';
    case 4:
        return s[0][idx+1]==' '&&s[2][idx]==' '&&s[2][idx+1]==' ';
    case 5:
        return s[1][idx+2]==' '&&s[2][idx]==' ';
    case 6:
        return s[1][idx+2]==' ';
    case 7:
        return s[1][idx]==' '&&s[1][idx+1]==' '&&s[2][idx]==' '&&s[2][idx+1]==' ';
    case 8:
        return true;
    case 9:
        return s[2][idx]==' ';
    default:
        return false;
    }
}

bool check_time(int hour, int minute, int idx){
    return check_num(hour/10, idx) && check_num(hour%10, idx+3) && check_num(minute/10, idx+6) && check_num(minute%10, idx+9);
}

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int t;
    cin>>t;
    getline(cin, s[0]);
    while(t--){
        for(int i=0;i<3;++i){
            getline(cin, s[i]);
        }
        int ans = -1;

        for(int hour=0;hour<24;++hour){
            for(int minute=0;minute<59;++minute){
                int delayminute = minute + 15;
                int delayhour = hour + (delayminute / 60);
                delayminute %= 60;
                delayhour %= 24;
                if(check_time(hour, minute, 0) && check_time(delayhour, delayminute, 13)){
                    if(ans == -1){
                        ans = hour * 100 + minute;
                    }
                    else{
                        ans = -2;
                    }
                }
            }
        }
        if(ans >= 0){
            printf("%04d\n", ans);
        }
        else{
            printf("Not Sure\n");
        }
    }
}
```



## F. Sorting It All Out

`Sortings`

```c++
#include<bits/stdc++.h>
using namespace std;

int edge[26][26];
int indeg[26], outdeg[26];
int n, m;
bool is_unique, no_circle;

string test(){
    memset(indeg, 0, sizeof(indeg));
    memset(outdeg, 0, sizeof(outdeg));
    is_unique = true;
    no_circle = true;
    string res;

    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            if(edge[i][j]){
                ++outdeg[i];
                ++indeg[j];
            }
        }
    }

    while(true){
        int tmp = -1;
        for(int i=0;i<n;++i){
            if(indeg[i] == 0){
                if(tmp == -1){
                    tmp = i;
                }
                else{
                    is_unique=false;
                }
            }
        }
        if(tmp == -1){
            break;
        }
        res.push_back('A'+tmp);
        indeg[tmp] = -1;
        for(int i=0;i<n;++i){
            if(edge[tmp][i]){
                --indeg[i];
            }
        }
    }

    for(int i=0;i<n;++i){
        if(indeg[i] != -1){
            no_circle = false;
        }
    }
    return res;
}

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    while(cin>>n>>m, n && m){
        bool finished = false;
        memset(edge, 0, sizeof(edge));
        for(int i=0;i<m;++i){
            char a, c, b;
            cin>>a>>c>>b;
            if(finished){
                continue;
            }
            edge[a-'A'][b-'A'] = 1;
            string res = test();
            if(!no_circle){
                finished = true;
                cout<<"Inconsistency found after "<<i+1<<" relations."<<endl;
            }
            if(no_circle && is_unique){
                finished = true;
                cout<<"Sorted sequence determined after "<<i+1<<" relations: "<<res<<"."<<endl;
            }
        }
        if(!finished){
            cout<<"Sorted sequence cannot be determined."<<endl;
        }
    }
}
```



## G. Rails

`Stack`

```c++
#include<bits/stdc++.h>
using namespace std;

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    stack<int> s;
    int n;
    while(cin>>n, n){
        while(true){
            bool impossible = false;
            int cnt = 1;
            while(!s.empty()){
                s.pop();
            }
            for(int i=0;i<n;++i){
                int tmp;
                cin>>tmp;
                if(i==0 && tmp == 0){
                    goto end_loop;
                }
                if(impossible){
                    continue;
                }
                while(s.empty() || s.top() < tmp){
                    s.push(cnt);
                    ++cnt;
                }
                if(s.top() == tmp){
                    s.pop();
                }
                else{
                    impossible = true;
                }
            }
            cout<<(impossible? "No":"Yes")<<endl;
        }
        end_loop:
            cout<<endl;
    }
}
```



##  H. The Rotation Game

`IDA*`

```c++
#include<bits/stdc++.h>
using namespace std;

int opposite[8] = { 5, 4, 7, 6, 1, 0, 3, 2 };
int maxdepth;
int line[8][7] = {
	{ 0, 2, 6, 11, 15, 20, 22 },
    { 1, 3, 8, 12, 17, 21, 23 },
    { 10, 9, 8, 7, 6, 5, 4 },
    { 19, 18, 17, 16, 15, 14, 13 },
    { 23, 21, 17, 12, 8, 3, 1 },
    { 22, 20, 15, 11, 6, 2, 0 },
    { 13, 14, 15, 16, 17, 18, 19 },
    { 4, 5, 6, 7, 8, 9, 10 }
};
string ans;
int circle[8] = { 6,7,8,11,12,15,16,17 };
int board[24];

int estimate()
{
	int cnt[4] = { 0 }, ans = 0;
	for (int i = 0; i<8; ++i)
		ans = max(ans, ++cnt[board[circle[i]]]);
	return 8 - ans;
}

void doit(int op)
{
  int temp = board[line[op][0]];
  for (int i = 0; i<6; ++i)
    board[line[op][i]] = board[line[op][i + 1]];
  board[line[op][6]] = temp;
}

bool dfs(int depth)
{
	int guess = estimate();
	if (guess == 0)
		return true;
	else if (depth + guess>maxdepth)
		return false;
	for (int i = 0; i<8; ++i)
	{
		doit(i);
		ans.push_back('A' + i);
		if (dfs(depth + 1))
			return true;
		ans.pop_back();
		doit(opposite[i]);
	}
	return false;
}

void ida()
{
	for (int i = 1;; ++i)
	{
		maxdepth = i;
		if (dfs(0))
			break;
	}
}

int main()
{
	while (cin >> board[0], board[0])
	{
		ans.clear();
		for (int i = 1; i<24; ++i)
			cin >> board[i];
		if (estimate() == 0)
		{
			cout << "No moves needed" << endl;
			cout << board[7] << endl;
		}
		else
		{
			ida();
			cout << ans << endl;
			cout << board[7] << endl;
		}
	}
}
```

