# Use Colab

I use [Colab](https://colab.research.google.com/) as the experiment platform.

Upload the Face Place(see [face_db](https://github.com/Cugtyt/IdentityRecognition/blob/master/face_db.md) for detail) to Google Drive.

To use the data in Colab, the code below is needed:

``` jupyter
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
```

Then follow the instruction of the output, after this step done, run the code below:

``` jupyter
!mkdir -p drive
!google-drive-ocamlfuse drive
```

You will see drive folder, by using:

``` jupyter
!ls
```

Then you can find the database you uploaded in that folder, finally you can play with algo.
