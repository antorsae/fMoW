basename=$(basename $1)
aws s3 cp $1 s3://antor-fmow/$basename --acl public-read
cat << EOF
public class FunctionalMap  {
  public String getAnswerURL() {
    //Replace the returned String with your submission file's URL
    return "https://antor-fmow.s3.amazonaws.com/$basename";
  }
}
EOF

